import unittest
from functools import partial
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from nnunetv2.experiment_planning.experiment_planners.mobile_unet_planner import (
    BraTSPlanner,
    MobileUNetPlanner,
)
from nnunetv2.experiment_planning.experiment_planners.stemmed_planner import StemmedPlanner
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.network_architecture.mobile_unet import MobileUNet
from nnunetv2.network_architecture.nd import InstanceNormNd
from nnunetv2.network_architecture.uib import InvertedBottleneckBlock
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.training.data_augmentation.custom_transforms.decoupled_spatial import (
    PairedMaskImageTransform,
    PairedSpatialTransform,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.split_resolution import SplitResolutionGeometry, normalize_factor
from nnunetv2.training.dataloading.split_resolution_data_loader import SplitResolutionDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdamW import nnUNetTrainerAdamW


class DummyPlanner(StemmedPlanner):
    def __init__(self):
        self.overwrite_target_spacing = None
        self.preprocessor_name = "DefaultPreprocessor"
        self.UNet_min_batch_size = 2
        self.plans_identifier = "StemmedPlans"
        self.dataset_fingerprint = {
            "spacings": [[1.0, 2.0, 3.0]],
            "shapes_after_crop": [[96, 96, 96]],
        }


class SplitResolutionTests(unittest.TestCase):
    percentiles = {str(p): [1.0, 2.0, 3.0] for p in range(0, 101, 5)}

    def test_exact_geometry_handles_fractional_anisotropic_and_negative_coordinates(self):
        geometry = SplitResolutionGeometry.from_spacings([1.5, 2, 4], [1, 1, 2])
        self.assertEqual(geometry.output_scale, (Fraction(3, 2), Fraction(2), Fraction(2)))
        self.assertEqual(geometry.denominators, (2, 1, 1))
        self.assertEqual(geometry.align_input_origin([-3, -2, 5]), (-4, -2, 5))
        self.assertEqual(
            geometry.align_input_origin_bounds([-35, 0, 0], [-35, 3, 3]),
            ((-36, -36), (0, 3), (0, 3)),
        )
        self.assertEqual(geometry.input_boundary_to_target([-4, -2, 5]), (-6, -4, 10))
        self.assertEqual(geometry.map_input_crop((slice(2, 11), slice(0, 3), slice(1, 4))),
                         (slice(3, 17), slice(0, 6), slice(2, 8)))
        with self.assertRaisesRegex(ValueError, "not aligned"):
            geometry.input_boundary_to_target([1, 0, 0])

    def test_factor_validation(self):
        self.assertEqual(normalize_factor(1.5, 3), (Fraction(3, 2),) * 3)
        self.assertEqual(normalize_factor(1.5000000001, 3), (Fraction(3, 2),) * 3)
        self.assertEqual(normalize_factor([1, 1.5, 2], 3),
                         (Fraction(1), Fraction(3, 2), Fraction(2)))
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                normalize_factor(invalid, 3)
        with self.assertRaises(ValueError):
            normalize_factor([1, 2], 3)

    def test_planner_scales_percentiles_transposes_and_emits_hires_geometry(self):
        isotropic = {str(p): [1.0, 1.0, 1.0] for p in range(0, 101, 5)}
        planner = DummyPlanner()
        planner.dataset_fingerprint["spacings"] = [[1.0, 1.0, 1.0]]
        plan = DummyPlanner()._plan_for_preset(
            "1.5x+2x-hires", [2, 0, 1], isotropic, np.array([1.0, 1.0, 1.0])
        )
        self.assertEqual(plan["data_identifier"], "StemmedPlans_1.5x+2x-hires")
        self.assertTrue(plan["preserve_segmentation_resolution"])
        np.testing.assert_allclose(plan["segmentation_spacing"], np.asarray(plan["spacing"]) / 1.5)
        arch = plan["architecture"]["arch_kwargs"]
        self.assertEqual(arch["head_stride"], [int(value * 1.5) for value in arch["stem_stride"]])
        self.assertNotIn("head_stride", plan)

        class AnisotropicPlanner(DummyPlanner):
            presets = {"vector": {"prep_downsampling_factor": [2, 1, 2],
                                   "stem_downsampling_factor": 2, "num_stages": 4,
                                   "preserve_segmentation_resolution": True}}

        vector_plan = AnisotropicPlanner()._plan_for_preset(
            "vector", [2, 0, 1], self.percentiles, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(vector_plan["segmentation_spacing"],
                                   np.asarray(vector_plan["spacing"]) / [2, 2, 1])

    def test_non_integral_head_geometry_is_rejected(self):
        class BadPlanner(DummyPlanner):
            presets = {"bad": {"prep_downsampling_factor": 1.5,
                                "stem_downsampling_factor": 1, "num_stages": 1,
                                "preserve_segmentation_resolution": True}}

        with self.assertRaisesRegex(ValueError, "non-integral MobileUNet head stride"):
            BadPlanner()._plan_for_preset(
                "bad", [0, 1, 2], self.percentiles, np.array([1.0, 2.0, 3.0])
            )

    def test_inheritance_derives_target_patch_only_from_spacing(self):
        plans = {"dataset_name": "Dataset999_Test", "plans_name": "TestPlans", "configurations": {
            "base": {"spacing": [1.5, 2, 2], "segmentation_spacing": [1, 1, 2],
                     "patch_size_unit": [4, 4, 4], "patch_size_multiplier": None,
                     "architecture": {"network_class_name": "MobileUNet",
                                      "arch_kwargs": {"head_stride": [99, 99, 99]}}},
            "child": {"inherits_from": "base", "patch_size_multiplier": 2},
        }}
        configuration = PlansManager(plans).get_configuration("child")
        self.assertEqual(configuration.patch_size, [8, 8, 8])
        self.assertEqual(configuration.segmentation_patch_size, [12, 16, 8])
        self.assertEqual(configuration.segmentation_spacing, [1, 1, 2])

    def test_mobile_hires_presets_include_brats_research_planner(self):
        planner = object.__new__(MobileUNetPlanner)
        self.assertIn("MN-1.5x+2x-hires-S", planner.configs)
        self.assertTrue(issubclass(BraTSPlanner, MobileUNetPlanner))

    def test_paired_transform_is_stateless(self):
        transform = PairedSpatialTransform(
            (8, 8, 8), 0, False, p_rotation=0, p_scaling=1, scaling=1.0,
            bg_style_seg_sampling=False, mode_seg="nearest", border_mode_seg="constant",
            padding_value_seg=-1, patch_size_seg=(12, 16, 8),
        )
        image = torch.zeros((1, 10, 10, 10))
        target = torch.zeros((1, 15, 20, 10), dtype=torch.int16)
        before = transform.patch_size
        result = transform(image=image, segmentation=target)
        self.assertEqual(transform.patch_size, before)
        self.assertEqual(result["image"].shape, (1, 8, 8, 8))
        self.assertEqual(result["segmentation"].shape, (1, 12, 16, 8))

    def test_paired_mask_maps_high_resolution_mask_to_image(self):
        transform = PairedMaskImageTransform([0], 0, 0)
        image = torch.ones((1, 2, 2, 2))
        segmentation = torch.zeros((1, 3, 4, 2), dtype=torch.int16)
        segmentation[:, :2] = -1
        result = transform(image=image, segmentation=segmentation)
        self.assertTrue((result["image"] == 0).any())
        self.assertTrue((result["image"] == 1).any())

    def test_preprocessing_keeps_foreground_locations_in_target_coordinates(self):
        class NoNormalizationPreprocessor(DefaultPreprocessor):
            def _normalize(self, data, seg, configuration_manager, intensity_properties):
                return data

        class LabelManager:
            has_regions = False
            foreground_regions = None
            foreground_labels = [1]
            has_ignore_label = False

        plans = SimpleNamespace(
            transpose_forward=[0, 1, 2],
            foreground_intensity_properties_per_channel={},
            get_label_manager=lambda _: LabelManager(),
        )
        resample_data = partial(resample_data_or_seg_to_shape, is_seg=False, order=1,
                                order_z=0, force_separate_z=False)
        resample_seg = partial(resample_data_or_seg_to_shape, is_seg=True, order=0,
                               order_z=0, force_separate_z=False)
        common = dict(resampling_fn_data=resample_data, resampling_fn_seg=resample_seg,
                      use_mask_for_norm=[False])
        split = SimpleNamespace(spacing=[2, 2, 2], segmentation_spacing=[1, 1, 1], **common)
        equal = SimpleNamespace(spacing=[2, 2, 2], segmentation_spacing=[2, 2, 2], **common)
        data = np.ones((1, 8, 8, 8), dtype=np.float32)
        segmentation = np.zeros((1, 8, 8, 8), dtype=np.int16)
        segmentation[0, 6, 6, 6] = 1
        preprocessor = NoNormalizationPreprocessor(verbose=False)
        split_data, split_seg, properties = preprocessor.run_case_npy(
            data, segmentation, {"spacing": [1, 1, 1]}, plans, split,
            {"labels": {"background": 0, "foreground": 1}})
        equal_data, equal_seg, _ = preprocessor.run_case_npy(
            data, segmentation, {"spacing": [1, 1, 1]}, plans, equal,
            {"labels": {"background": 0, "foreground": 1}})
        self.assertEqual(split_data.shape, (1, 4, 4, 4))
        self.assertEqual(split_seg.shape, (1, 8, 8, 8))
        self.assertEqual(equal_data.shape, equal_seg.shape)
        location = properties["class_locations"][1][0]
        self.assertEqual(len(location), 4)
        self.assertTrue(np.all(location[1:] < np.asarray(split_seg.shape[1:])))

    def test_blosc_segmentation_chunks_use_target_patch_size(self):
        class SavingPreprocessor(DefaultPreprocessor):
            def run_case(self, *args, **kwargs):
                return (np.zeros((1, 4, 4, 4), dtype=np.float32),
                        np.zeros((1, 6, 8, 4), dtype=np.int16), {})

        configuration = SimpleNamespace(patch_size=[4, 4, 4],
                                        segmentation_patch_size=[6, 8, 4])
        seen = []

        def fake_params(shape, patch_size, itemsize):
            seen.append(tuple(patch_size))
            return tuple(shape), tuple(shape)

        with patch("nnunetv2.preprocessing.preprocessors.default_preprocessor.comp_blosc2_params",
                   side_effect=fake_params), patch.object(nnUNetDatasetBlosc2, "save_case"):
            SavingPreprocessor(verbose=False).run_case_save(
                "unused", [], "unused", None, configuration, {})
        self.assertEqual(seen, [(4, 4, 4), (6, 8, 4)])

    def test_split_loader_maps_foreground_and_crop_boundaries_exactly(self):
        geometry = SplitResolutionGeometry.from_spacings([1.5, 2, 1], [1, 1, 1])

        class Dataset:
            identifiers = ["case"]

            @staticmethod
            def load_case(_):
                data = np.zeros((1, 8, 8, 8), dtype=np.float32)
                seg = np.zeros((1, 12, 16, 8), dtype=np.int16)
                seg[0, 7, 9, 4] = 1
                return data, seg, None, {"class_locations": {1: np.array([[0, 7, 9, 4]])}}

        label_manager = SimpleNamespace(has_ignore_label=False, all_labels=[0, 1])
        loader = SplitResolutionDataLoader(
            Dataset(), 1, (4, 4, 4), (4, 4, 4), label_manager,
            oversample_foreground_percent=1, transforms=None,
            segmentation_patch_size=(6, 8, 4), geometry=geometry)
        batch = loader.generate_train_batch()
        self.assertEqual(tuple(batch["data"].shape), (1, 1, 4, 4, 4))
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 6, 8, 4))
        self.assertTrue((batch["target"] == 1).any())

    def test_split_loader_adds_virtual_padding_for_unaligned_singleton_origin(self):
        geometry = SplitResolutionGeometry.from_spacings([1.5] * 3, [1] * 3)

        class Dataset:
            identifiers = ["case"]

            @staticmethod
            def load_case(_):
                return (
                    np.ones((1, 2, 2, 2), dtype=np.float32),
                    np.ones((1, 3, 3, 3), dtype=np.int16),
                    None,
                    {"class_locations": {}},
                )

        label_manager = SimpleNamespace(has_ignore_label=False, all_labels=[0, 1])
        loader = SplitResolutionDataLoader(
            Dataset(), 1, (8, 8, 8), (8, 8, 8), label_manager,
            transforms=None, segmentation_patch_size=(12, 12, 12), geometry=geometry,
        )
        batch = loader.generate_train_batch()
        self.assertEqual(tuple(batch["data"].shape), (1, 1, 8, 8, 8))
        self.assertEqual(tuple(batch["target"].shape), (1, 1, 12, 12, 12))

    def test_mobile_forward_matches_high_resolution_target(self):
        network = MobileUNet(
            ndim=3, input_channels=1, num_classes=2, num_stages=2,
            stem_kernel_size=[3] * 3, stem_stride=[2] * 3,
            head_kernel_size=[5] * 3, head_stride=[3] * 3,
            channels=[4, 8], encoder_expansion_ratios=[1, 1], decoder_expansion_ratios=[1],
            kernel_sizes=[[3] * 3] * 2, strides=[[1] * 3, [2] * 3],
            norm_layer=InstanceNormNd, norm_kwargs={}, act_layer=torch.nn.ReLU,
            act_kwargs={"inplace": True}, encoder_depths=[1, 1], decoder_depths=[1],
            block_factory=InvertedBottleneckBlock, deep_supervision=False)
        output = network(torch.randn(1, 1, 16, 16, 16))
        target = torch.zeros((1, 1, 24, 24, 24), dtype=torch.long)
        self.assertEqual(output.shape[2:], target.shape[2:])

    def test_split_capability_checks_and_standard_transform_path(self):
        invalid = SimpleNamespace(
            configuration_manager=SimpleNamespace(
                split_resolution_geometry=SplitResolutionGeometry.from_spacings([2] * 3, [1] * 3),
                patch_size=[16] * 3,
                network_arch_class_name="not.MobileUNet",
            ),
            is_cascaded=True,
            enable_deep_supervision=True,
            configure_rotation_dummyDA_mirroring_and_inital_patch_size=lambda: (None, True, None, None),
        )
        with self.assertRaisesRegex(RuntimeError, "MobileUNet.*cascades.*deep_supervision.*dummy-2D"):
            nnUNetTrainerAdamW._validate_split_resolution_configuration(invalid)
        self.assertFalse(nnUNetTrainer.supports_split_resolution)
        self.assertTrue(nnUNetTrainerAdamW.supports_split_resolution)

        transforms = nnUNetTrainerAdamW.get_training_transforms(
            patch_size=(8, 8, 8), rotation_for_DA=(-0.1, 0.1), deep_supervision_scales=None,
            mirror_axes=(0, 1, 2), do_dummy_2d_data_aug=False, use_mask_for_norm=[False])
        self.assertFalse(any(isinstance(item, PairedSpatialTransform) for item in transforms.transforms))

    def test_sliding_window_fractional_output_shape_and_coverage(self):
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=False, use_mirroring=False,
                                    perform_everything_on_device=False, device=torch.device("cpu"),
                                    allow_tqdm=False)
        predictor.configuration_manager = SimpleNamespace(
            patch_size=[4, 4, 4], segmentation_patch_size=[6, 6, 6],
            split_resolution_geometry=SplitResolutionGeometry.from_spacings([1.5] * 3, [1] * 3))
        predictor.label_manager = SimpleNamespace(num_segmentation_heads=1)
        class ConstantNetwork(torch.nn.Module):
            def forward(self, value):
                return torch.ones((value.shape[0], 1, 6, 6, 6), device=value.device)

        predictor.network = ConstantNetwork()
        result = predictor.predict_sliding_window_return_logits(torch.ones((1, 9, 7, 5)))
        self.assertEqual(result.shape, (1, 14, 11, 8))
        self.assertTrue(torch.all(result == 1))
        predictor.use_mirroring = True
        predictor.allowed_mirroring_axes = (0, 1, 2)
        mirrored = predictor.predict_sliding_window_return_logits(torch.ones((1, 9, 7, 5)))
        self.assertTrue(torch.equal(result, mirrored))
        predictor.use_gaussian = True
        gaussian = predictor.predict_sliding_window_return_logits(torch.ones((1, 9, 7, 5)))
        self.assertTrue(torch.allclose(gaussian.float(), torch.ones_like(gaussian.float()), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
