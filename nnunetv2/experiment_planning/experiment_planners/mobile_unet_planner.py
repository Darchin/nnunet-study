from copy import deepcopy
from typing import List, Tuple, Union

from nnunetv2.experiment_planning.experiment_planners.stemmed_planner import (
    StemmedPlanner,
)


class MobileUNetPlanner(StemmedPlanner):
    trainer_defaults = {
        "initial_lr": 3e-4,
        "weight_decay": 1e-3,
        "num_epochs": 250,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "enable_deep_supervision": False,
    }
    mobile_presets = {
        "mn-2x": {
            "inherits_from": "2x",
            "encoder_depths": [2, 3, 3, 9, 3],
            "decoder_depths": [1, 1, 1, 1],
            "encoder_expansion_ratios": [2.0, 2.0, 4.0, 4.0, 4.0],
            "decoder_expansion_ratios": 1.0,
        },
        "mn-3x": {
            "inherits_from": "3x",
            "encoder_depths": [3, 3, 9, 3],
            "decoder_depths": [1, 1, 1],
            "encoder_expansion_ratios": [2.0, 4.0, 4.0, 4.0],
            "decoder_expansion_ratios": 1.0,
        },
        "mn-4x": {
            "inherits_from": "4x",
            "encoder_depths": [3, 3, 9, 3],
            "decoder_depths": [1, 1, 1],
            "encoder_expansion_ratios": [2.0, 4.0, 4.0, 4.0],
            "decoder_expansion_ratios": 1.0,
        },
    }

    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = 8,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "MobileUNetPlans",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing,
            suppress_transpose,
        )

    @classmethod
    def _mobile_configuration(cls, preset: dict) -> dict:
        parent_name = preset["inherits_from"]
        num_stages = cls.presets[parent_name]["num_stages"]
        dim = 3
        return {
            "inherits_from": [parent_name, "mn"],
            "architecture": {
                "network_class_name": "nnunetv2.network_architecture.mobile_unet.MobileUNet",
                "arch_kwargs": {
                    "ndim": dim,
                    "kernel_sizes": [[3] * dim for _ in range(num_stages)],
                    "strides": [[1] * dim]
                    + [[2] * dim for _ in range(num_stages - 1)],
                    "norm_layer": "nnunetv2.network_architecture.nd.InstanceNormNd",
                    "norm_kwargs": {},
                    "act_layer": "torch.nn.ReLU",
                    "act_kwargs": {"inplace": True},
                    "encoder_depths": preset["encoder_depths"],
                    "decoder_depths": preset["decoder_depths"],
                    "encoder_expansion_ratios": preset[
                        "encoder_expansion_ratios"
                    ],
                    "decoder_expansion_ratios": preset[
                        "decoder_expansion_ratios"
                    ],
                },
                "_kw_requires_import": ("norm_layer", "act_layer"),
            },
            "required_for_training": [
                "patch_size_multiplier",
                "architecture.network_class_name",
                "architecture.arch_kwargs.channels",
            ],
        }

    def _additional_configurations(self) -> dict:
        configurations = {"mn": {"trainer": dict(self.trainer_defaults)}}
        configurations.update(
            {
                name: self._mobile_configuration(preset)
                for name, preset in self.mobile_presets.items()
            }
        )
        return configurations


class BaselinePlanner(MobileUNetPlanner):
    _presets = {
        "mn-2x-t": {
            "inherits_from": "mn-2x",
            "patch_size_multiplier": 6,
            "channels": [16, 32, 64, 128, 256],
        },
        "mn-2x-s": {
            "inherits_from": "mn-2x",
            "patch_size_multiplier": 6,
            "channels": [32, 64, 128, 256, 512],
        },
        "mn-2x-m": {
            "inherits_from": "mn-2x",
            "patch_size_multiplier": 6,
            "channels": [48, 96, 192, 384, 768],
        },
        "mn-3x-t": {
            "inherits_from": "mn-3x",
            "patch_size_multiplier": 8,
            "channels": [32, 64, 128, 256],
        },
        "mn-3x-s": {
            "inherits_from": "mn-3x",
            "patch_size_multiplier": 8,
            "channels": [64, 128, 256, 512],
        },
        "mn-3x-m": {
            "inherits_from": "mn-3x",
            "patch_size_multiplier": 8,
            "channels": [96, 192, 384, 768],
        },
        "mn-4x-t": {
            "inherits_from": "mn-4x",
            "patch_size_multiplier": 6,
            "channels": [32, 64, 128, 256],
        },
        "mn-4x-s": {
            "inherits_from": "mn-4x",
            "patch_size_multiplier": 6,
            "channels": [64, 128, 256, 512],
        },
        "mn-4x-m": {
            "inherits_from": "mn-4x",
            "patch_size_multiplier": 6,
            "channels": [96, 192, 384, 768],
        },
    }

    @classmethod
    def _configuration(cls, preset: dict) -> dict:
        return {
            "inherits_from": preset["inherits_from"],
            "patch_size_multiplier": preset["patch_size_multiplier"],
            "architecture": {
                "arch_kwargs": {
                    "channels": preset["channels"],
                },
            },
        }

    def _additional_configurations(self) -> dict:
        configurations = super()._additional_configurations()
        configurations.update(
            {
                name: self._configuration(preset)
                for name, preset in self._presets.items()
            }
        )
        return configurations


class TemporaryPlanner(BaselinePlanner):
    _temporary_configurations = {}

    def _additional_configurations(self) -> dict:
        configurations = super()._additional_configurations()
        configurations.update(deepcopy(self._temporary_configurations))
        return configurations
