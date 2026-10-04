"""AdamW configuration, loader selection, and batch-worker compatibility."""

import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.run import batch_train
from nnunetv2.training.dataloading.shared_batch_augmenter import SharedBatchAugmenter
from nnunetv2.training.data_augmentation.custom_transforms.fast_cpu import (
    FastGaussianBlurTransform,
    FastRegionsTransform,
)
from nnunetv2.training.data_augmentation.custom_transforms.fast_spatial import FastSpatialTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdamW import nnUNetTrainerAdamW

module = importlib.import_module(nnUNetTrainerAdamW.__module__)


class Checks(unittest.TestCase):
    def trainer(self, configuration):
        trainer = nnUNetTrainerAdamW.__new__(nnUNetTrainerAdamW)
        trainer.configuration_manager = SimpleNamespace(trainer=configuration)
        trainer.device = torch.device("cuda")
        with patch.object(nnUNetTrainer, "__init__", return_value=None):
            trainer.__init__({}, "test", 0, {}, torch.device("cuda"))
        return trainer

    def test_existing_configuration_and_pinning_precedence(self):
        config = dict(
            initial_lr=0.001,
            weight_decay=0.02,
            num_epochs=9,
            warmup_epochs=3,
            min_lr=0.00001,
            enable_deep_supervision=True,
            pin_memory=False,
            router_schedule=dict(start_val=2.0, end_val=0.5, end_epoch=7),
        )
        trainer = self.trainer(config)
        for key in set(config) - {"router_schedule"}:
            self.assertEqual(getattr(trainer, key), config[key])
        self.assertEqual(trainer.router_scheduler.get_value(7), 0.5)
        with patch.dict("os.environ", nnUNet_pin_memory="true"):
            self.assertFalse(trainer._should_pin_memory())
        trainer.pin_memory = None
        with patch.dict("os.environ", nnUNet_pin_memory="false"):
            self.assertFalse(trainer._should_pin_memory())

    def test_invalid_configuration_still_fails(self):
        for config, error in [
            (dict(pin_memory="true"), TypeError),
            (dict(num_epochs=True), TypeError),
            (dict(warmup_epochs=1), ValueError),
            (dict(num_epochs=5, warmup_epochs=5), ValueError),
            (dict(unknown_parameter=1), ValueError),
        ]:
            with self.subTest(config=config), self.assertRaises(error):
                self.trainer(config)

    def test_real_transform_tree_preserves_dummy_2d_and_deep_supervision(self):
        def members(transform):
            yield transform
            for child in getattr(transform, "transforms", []):
                yield from members(child)
            if hasattr(transform, "transform"):
                yield from members(transform.transform)

        for dummy in [False, True]:
            transforms = nnUNetTrainerAdamW.get_training_transforms(
                (32, 48, 64),
                (-0.5, 0.5),
                [(1, 1, 1), (0.5, 0.5, 0.5)],
                (0, 1, 2),
                dummy,
                regions=[(1, 2), 2],
                foreground_labels=[1, 2],
                ignore_label=None,
            )
            found = list(members(transforms))
            self.assertEqual(sum(isinstance(t, FastSpatialTransform) for t in found), 1)
            self.assertEqual(sum(isinstance(t, FastGaussianBlurTransform) for t in found), 1)
            self.assertEqual(sum(isinstance(t, FastRegionsTransform) for t in found), 1)
            self.assertEqual(
                next(t for t in found if isinstance(t, FastSpatialTransform)).patch_size,
                (48, 64) if dummy else (32, 48, 64),
            )

    def test_loader_selection_respects_device_pin_and_zero_workers(self):
        for workers, pin, device, expected in [
            (0, True, "cuda", SingleThreadedAugmenter),
            (2, True, "cuda", SharedBatchAugmenter),
            (2, False, "cuda", NonDetMultiThreadedAugmenter),
            (2, False, "cpu", NonDetMultiThreadedAugmenter),
        ]:
            with self.subTest(workers=workers, pin=pin, device=device):
                trainer = self.trainer(dict(pin_memory=pin))
                trainer.device = torch.device(device)
                trainer.dataset_class = Mock()
                trainer.batch_size = 2
                trainer.configuration_manager.patch_size = (16, 16, 16)
                trainer.configuration_manager.use_mask_for_norm = [False]
                trainer.is_cascaded = False
                trainer.label_manager = SimpleNamespace(foreground_labels=[1], has_regions=False, ignore_label=None)
                trainer.oversample_foreground_percent = 0.33
                trainer.probabilistic_oversampling = False
                trainer._get_deep_supervision_scales = Mock(return_value=None)
                trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size = Mock(
                    return_value=((-1, 1), False, (24, 24, 24), (0, 1, 2))
                )
                trainer.get_tr_and_val_datasets = Mock(return_value=(Mock(), Mock()))
                with (
                    patch.object(module, "get_allowed_n_proc_DA", return_value=workers),
                    patch.object(module, "nnUNetDataLoader"),
                    patch.object(expected, "__next__", return_value={}),
                ):
                    train, val = trainer.get_dataloaders()
                    self.assertIs(type(train), expected)
                    self.assertIs(type(val), expected)
                    self.assertIsNone(getattr(train, "pool_loader", None))

    def test_batch_worker_preserves_trainer_and_training_options(self):
        trainer = Mock()
        with (
            patch.object(batch_train, "load_json", return_value={}),
            patch.object(batch_train, "load_dataset_json_for_plans", return_value={}),
            patch.object(
                batch_train, "recursive_find_trainer_class_by_name", return_value=Mock(return_value=trainer)
            ) as lookup,
            patch.object(batch_train, "maybe_load_checkpoint") as checkpoint,
            patch.object(batch_train.torch, "set_num_threads"),
            patch.object(batch_train.torch, "set_num_interop_threads"),
            patch.object(batch_train.torch.cuda, "is_available", return_value=False),
        ):
            batch_train.run_training_worker_process(
                0, 1, "plans.json", "config", 2, "nnUNetTrainerAdamW", True, 7, True, True
            )
        lookup.assert_called_once_with("nnUNetTrainerAdamW")
        self.assertEqual(trainer.checkpoint_interval, 7)
        self.assertTrue(trainer.disable_train_val)
        checkpoint.assert_called_once_with(
            trainer, continue_training=True, validation_only=False, pretrained_weights_file=None
        )
        trainer.run_training.assert_called_once_with()
        trainer.perform_actual_validation.assert_called_once_with(False, False)
        command = batch_train.make_worker_command(
            "plans.json", "config", 2, "nnUNetTrainerAdamW", True, 7, True, continue_training=True
        )
        for flag in ["--disable-tta", "--disable_train_val", "--c"]:
            self.assertIn(flag, command)
