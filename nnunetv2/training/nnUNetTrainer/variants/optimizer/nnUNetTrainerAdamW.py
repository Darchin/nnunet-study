import functools
import math
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

from nnunetv2.network_architecture.moe import Router
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class class_or_instance_method:
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __get__(self, instance, owner):
        if instance is None:
            def class_call(*args, **kwargs):
                return self.fn(owner, *args, **kwargs)
            return class_call

        def instance_call(*args, **kwargs):
            return self.fn(instance, *args, **kwargs)
        return instance_call


class LinearWarmupCosineAnnealingLR:
    def __init__(self, optimizer, initial_lr: float, warmup_epochs: int, num_epochs: int, min_lr: float):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        self.min_lr = min_lr
        self.ctr = 0
        self._last_lr = [self._compute_lr(0) for _ in optimizer.param_groups]

        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr

    def _compute_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            progress = epoch / (self.warmup_epochs - 1)
            return self.initial_lr * (0.1 + 0.9 * progress)

        progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
        progress = min(max(progress, 0), 1)
        return self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self._compute_lr(current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class LinearRouterTemperatureScheduler:
    def __init__(self, start_val: float, end_val: float, end_epoch: int):
        self.start_val = start_val
        self.end_val = end_val
        self.end_epoch = end_epoch

    def get_value(self, epoch: int) -> float:
        progress = min(max(epoch / self.end_epoch, 0), 1)
        return self.start_val + (self.end_val - self.start_val) * progress


class nnUNetTrainerAdamW(nnUNetTrainer):
    configurable_trainer_keys = {
        'initial_lr',
        'weight_decay',
        'num_epochs',
        'warmup_epochs',
        'min_lr',
        'enable_deep_supervision',
        'router_schedule',
        '2d_aug',
        'use_nn_seg_resample',
    }

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 3e-4
        self.weight_decay = 1e-3
        self.num_epochs = 250
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        self.enable_deep_supervision = False
        self.two_d_aug = None
        self.use_nn_seg_resample = False
        self.router_scheduler = None
        self._apply_trainer_configuration()

    @staticmethod
    def _require_real(value, name: str, min_value: float = None, allow_zero: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"trainer.{name} must be a number, got {type(value).__name__}")
        value = float(value)
        if min_value is not None:
            if allow_zero:
                valid = value >= min_value
            else:
                valid = value > min_value
            if not valid:
                cmp = ">=" if allow_zero else ">"
                raise ValueError(f"trainer.{name} must be {cmp} {min_value}, got {value}")
        return value

    @staticmethod
    def _require_int(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"trainer.{name} must be an integer, got {type(value).__name__}")
        return value

    def _apply_trainer_configuration(self):
        if '2d_aug' in self.configuration_manager.configuration:
            val = self.configuration_manager.configuration['2d_aug']
            if val is not None and not isinstance(val, bool):
                raise TypeError(
                    f"configuration.2d_aug must be a bool or None, got {type(val).__name__}"
                )
            self.two_d_aug = val

        if 'use_nn_seg_resample' in self.configuration_manager.configuration:
            val = self.configuration_manager.configuration['use_nn_seg_resample']
            if not isinstance(val, bool):
                raise TypeError(
                    f"configuration.use_nn_seg_resample must be a bool, got {type(val).__name__}"
                )
            self.use_nn_seg_resample = val

        trainer_config = self.configuration_manager.trainer
        if not trainer_config:
            return
        if not isinstance(trainer_config, dict):
            raise TypeError(f"trainer must be a dict, got {type(trainer_config).__name__}")
        unknown_keys = set(trainer_config) - self.configurable_trainer_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown trainer configuration keys for {self.__class__.__name__}: {sorted(unknown_keys)}"
            )

        if '2d_aug' in trainer_config:
            val = trainer_config['2d_aug']
            if val is not None and not isinstance(val, bool):
                raise TypeError(
                    f"trainer.2d_aug must be a bool or None, got {type(val).__name__}"
                )
            self.two_d_aug = val

        if 'use_nn_seg_resample' in trainer_config:
            val = trainer_config['use_nn_seg_resample']
            if not isinstance(val, bool):
                raise TypeError(
                    f"trainer.use_nn_seg_resample must be a bool, got {type(val).__name__}"
                )
            self.use_nn_seg_resample = val

        if 'initial_lr' in trainer_config:
            self.initial_lr = self._require_real(trainer_config['initial_lr'], 'initial_lr', 0)
        if 'weight_decay' in trainer_config:
            self.weight_decay = self._require_real(trainer_config['weight_decay'], 'weight_decay', 0, allow_zero=True)
        if 'num_epochs' in trainer_config:
            self.num_epochs = self._require_int(trainer_config['num_epochs'], 'num_epochs')
        if 'warmup_epochs' in trainer_config:
            self.warmup_epochs = self._require_int(trainer_config['warmup_epochs'], 'warmup_epochs')
        if 'min_lr' in trainer_config:
            self.min_lr = self._require_real(trainer_config['min_lr'], 'min_lr', 0, allow_zero=True)
        if 'enable_deep_supervision' in trainer_config:
            if not isinstance(trainer_config['enable_deep_supervision'], bool):
                raise TypeError(
                    f"trainer.enable_deep_supervision must be a bool, got "
                    f"{type(trainer_config['enable_deep_supervision']).__name__}"
                )
            self.enable_deep_supervision = trainer_config['enable_deep_supervision']
        if trainer_config.get('router_schedule') is not None:
            router_schedule = trainer_config['router_schedule']
            if not isinstance(router_schedule, dict):
                raise TypeError(
                    f"trainer.router_schedule must be a dict or None, got {type(router_schedule).__name__}"
                )
            required_keys = {'start_val', 'end_val', 'end_epoch'}
            missing_keys = required_keys - set(router_schedule)
            unknown_keys = set(router_schedule) - required_keys
            if missing_keys:
                raise ValueError(
                    f"Missing trainer.router_schedule keys: {sorted(missing_keys)}"
                )
            if unknown_keys:
                raise ValueError(
                    f"Unknown trainer.router_schedule keys: {sorted(unknown_keys)}"
                )

            start_val = self._require_real(
                router_schedule['start_val'], 'router_schedule.start_val', 0
            )
            end_val = self._require_real(
                router_schedule['end_val'], 'router_schedule.end_val', 0
            )
            end_epoch = self._require_int(
                router_schedule['end_epoch'], 'router_schedule.end_epoch'
            )
            if end_epoch <= 0:
                raise ValueError(
                    f"trainer.router_schedule.end_epoch must be > 0, got {end_epoch}"
                )
            self.router_scheduler = LinearRouterTemperatureScheduler(
                start_val, end_val, end_epoch
            )

        if self.warmup_epochs < 2:
            raise ValueError(
                f"trainer.warmup_epochs must be >= 2 for the AdamW warmup schedule, got {self.warmup_epochs}"
            )
        if self.num_epochs <= self.warmup_epochs:
            raise ValueError(
                f"trainer.num_epochs must be greater than trainer.warmup_epochs, got "
                f"{self.num_epochs} and {self.warmup_epochs}"
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                      lr=self.initial_lr,
                                      weight_decay=self.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, self.initial_lr, self.warmup_epochs, self.num_epochs, self.min_lr
        )
        return optimizer, lr_scheduler

    def _update_router_temperatures(self):
        if self.router_scheduler is None:
            return

        temperature = self.router_scheduler.get_value(self.current_epoch)
        for module in self.network.modules():
            if isinstance(module, Router):
                for child in module.children():
                    if hasattr(child, 'temperature'):
                        # Keep the buffer object stable. Reassigning a Python
                        # scalar is observed by torch.compile as a changed
                        # guard and causes a costly recompile each epoch.
                        child.temperature.fill_(temperature)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._update_router_temperatures()

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        if self.two_d_aug is None:
            return super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            if self.two_d_aug:
                raise ValueError("2d_aug cannot be True for 2D configurations.")
            do_dummy_2d_data_aug = False
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            do_dummy_2d_data_aug = self.two_d_aug
            if do_dummy_2d_data_aug:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @class_or_instance_method
    def get_training_transforms(
        self_or_cls,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
        use_nn_seg_resample: bool = None,
    ) -> BasicTransform:
        if use_nn_seg_resample is None:
            use_nn_seg_resample = getattr(self_or_cls, 'use_nn_seg_resample', False)

        transforms = nnUNetTrainer.get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )

        target_mode_seg = 'nearest' if use_nn_seg_resample else 'bilinear'

        def _set_mode_seg(transform_list):
            for t in transform_list:
                if isinstance(t, SpatialTransform):
                    t.mode_seg = target_mode_seg
                elif hasattr(t, 'transforms') and isinstance(t.transforms, list):
                    _set_mode_seg(t.transforms)

        _set_mode_seg(transforms.transforms)

        if getattr(self_or_cls, 'log_file', None) is not None:
            self_or_cls.print_to_log_file(f'use_nn_seg_resample: {use_nn_seg_resample}')

        return transforms

