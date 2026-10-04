import math

import torch

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.data_augmentation.custom_transforms.fast_cpu import replace_cpu_transforms
from nnunetv2.training.data_augmentation.custom_transforms.fast_spatial import replace_spatial_transforms
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.shared_batch_augmenter import SharedBatchAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from nnunetv2.network_architecture.moe import Router
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


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
        'pin_memory',
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
        self.pin_memory = None
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

        if 'pin_memory' in trainer_config:
            val = trainer_config['pin_memory']
            if not isinstance(val, bool):
                raise TypeError(
                    f"trainer.pin_memory must be a bool, got {type(val).__name__}"
                )
            self.pin_memory = val

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

    def _should_pin_memory(self) -> bool:
        if self.pin_memory is not None:
            return self.pin_memory
        return super()._should_pin_memory()

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        transforms = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        return replace_spatial_transforms(replace_cpu_transforms(transforms))

    @staticmethod
    def get_validation_transforms(*args, **kwargs):
        return replace_cpu_transforms(nnUNetTrainer.get_validation_transforms(*args, **kwargs))

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch = self.configuration_manager.patch_size
        scales = self._get_deep_supervision_scales()
        rotation, dummy2d, initial, mirror = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        common = dict(is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
                      regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                      ignore_label=self.label_manager.ignore_label)
        train_transforms = self.get_training_transforms(
            patch, rotation, scales, mirror, dummy2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm, **common)
        val_transforms = self.get_validation_transforms(scales, **common)
        train_dataset, val_dataset = self.get_tr_and_val_datasets()
        loader_args = dict(
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            probabilistic_oversampling=self.probabilistic_oversampling)
        train_loader = nnUNetDataLoader(train_dataset, self.batch_size, initial, patch,
                                       transforms=train_transforms, **loader_args)
        val_loader = nnUNetDataLoader(val_dataset, self.batch_size, patch, patch,
                                     transforms=val_transforms, **loader_args)
        workers = get_allowed_n_proc_DA()
        pin = self._should_pin_memory()

        def wrap(loader, n, cached):
            if workers == 0:
                return SingleThreadedAugmenter(loader, None)
            if pin and self.device.type == 'cuda':
                return SharedBatchAugmenter(loader, n, cached)
            return NonDetMultiThreadedAugmenter(loader, None, n, cached,
                                               pin_memory=pin, wait_time=0.002)

        train = wrap(train_loader, workers, max(6, workers // 2))
        val = wrap(val_loader, max(1, workers // 2), max(3, workers // 4))
        try:
            next(train)
            next(val)
        except Exception:
            for augmenter in (train, val):
                if isinstance(augmenter, NonDetMultiThreadedAugmenter):
                    augmenter._finish(force=True)
            raise
        return train, val
