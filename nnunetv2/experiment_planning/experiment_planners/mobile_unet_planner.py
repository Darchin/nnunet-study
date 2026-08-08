from copy import deepcopy
from typing import List, Tuple, Union

from nnunetv2.experiment_planning.experiment_planners.stemmed_planner import (
    StemmedPlanner,
)


class MobileUNetPlanner(StemmedPlanner):
    _presets = {
        "mn": {
            "architecture": {
                "network_class_name": "nnunetv2.network_architecture.mobile_unet.MobileUNet",
                "arch_kwargs": {
                    "block_factory": "nnunetv2.network_architecture.uib.InvertedBottleneckBlock",
                    "norm_layer": "nnunetv2.network_architecture.nd.InstanceNormNd",
                    "norm_kwargs": {},
                    "act_layer": "torch.nn.ReLU",
                    "act_kwargs": {"inplace": True},
                },
                "_kw_requires_import": ("block_factory", "norm_layer", "act_layer"),
            },
            "required_for_training": [
                "ndim",
                "kernel_sizes",
                "strides",
                "patch_size_multiplier",
                "architecture.network_class_name",
                "architecture.arch_kwargs.channels",
                "architecture.arch_kwargs.encoder_depths",
                "architecture.arch_kwargs.decoder_depths",
                "architecture.arch_kwargs.encoder_expansion_ratios",
                "architecture.arch_kwargs.decoder_expansion_ratios",
            ],
            "trainer": {
                "initial_lr": 3e-4,
                "weight_decay": 1e-3,
                "num_epochs": 500,
                "warmup_epochs": 5,
                "min_lr": 1e-6,
                "enable_deep_supervision": False,
            },
        },
        "mn-2x": {
            "inherits_from": ["2x", "mn"],
            "arch_kwargs": {
                "ndim": 3,
                "kernel_sizes": [[3] * 3 for _ in range(5)],
                "strides": [[1] * 3] + [[2] * 3 for _ in range(5 - 1)],
                "encoder_depths": [2, 3, 3, 9, 3],
                "decoder_depths": [1, 1, 1, 1],
                "encoder_expansion_ratios": [2.0, 2.0, 4.0, 4.0, 4.0],
                "decoder_expansion_ratios": 1.0,
            },
        },
        "mn-3x": {
            "inherits_from": ["3x", "mn"],
            "architecture": {
                "arch_kwargs": {
                    "ndim": 3,
                    "kernel_sizes": [[3] * 3 for _ in range(4)],
                    "strides": [[1] * 3] + [[2] * 3 for _ in range(4 - 1)],
                    "encoder_depths": [3, 3, 9, 3],
                    "decoder_depths": [1, 1, 1],
                    "encoder_expansion_ratios": [2.0, 4.0, 4.0, 4.0],
                    "decoder_expansion_ratios": 1.0,
                }
            },
        },
        "mn-4x": {
            "inherits_from": ["4x", "mn"],
            "architecture": {
                "arch_kwargs": {
                    "ndim": 3,
                    "kernel_sizes": [[3] * 3 for _ in range(4)],
                    "strides": [[1] * 3] + [[2] * 3 for _ in range(4 - 1)],
                    "encoder_depths": [3, 3, 9, 3],
                    "decoder_depths": [1, 1, 1],
                    "encoder_expansion_ratios": [2.0, 4.0, 4.0, 4.0],
                    "decoder_expansion_ratios": 1.0,
                }
            },
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_presets" in cls.__dict__:
            merged = {}
            for base in reversed(cls.__mro__):
                if "_presets" in base.__dict__:
                    merged.update(deepcopy(base.__dict__["_presets"]))
            cls._presets = merged

    def _additional_configurations(self) -> dict:
        return deepcopy(self._presets)


class BaselinePlanner(MobileUNetPlanner):
    _presets = {
        "mn-2x-t": {
            "inherits_from": "mn-2x",
            "patch_size_multiplier": 6,
            "architecture": {"arch_kwargs": {"channels": [16, 32, 64, 128, 256]}},
        },
        "mn-2x-s": {
            "inherits_from": "mn-2x",
            "patch_size_multiplier": 6,
            "architecture": {"arch_kwargs": {"channels": [32, 64, 128, 256, 512]}},
        },
        "mn-2x-m": {
            "inherits_from": "mn-2x",
            "patch_size_multiplier": 6,
            "architecture": {
                "arch_kwargs": {
                    "channels": [48, 96, 192, 384, 768],
                }
            },
        },
        "mn-3x-t": {
            "inherits_from": "mn-3x",
            "patch_size_multiplier": 8,
            "architecture": {
                "arch_kwargs": {
                    "channels": [32, 64, 128, 256],
                }
            },
        },
        "mn-3x-s": {
            "inherits_from": "mn-3x",
            "patch_size_multiplier": 8,
            "architecture": {
                "arch_kwargs": {
                    "channels": [64, 128, 256, 512],
                }
            },
        },
        "mn-3x-m": {
            "inherits_from": "mn-3x",
            "patch_size_multiplier": 8,
            "architecture": {
                "arch_kwargs": {
                    "channels": [96, 192, 384, 768],
                }
            },
        },
        "mn-4x-t": {
            "inherits_from": "mn-4x",
            "patch_size_multiplier": 6,
            "architecture": {
                "arch_kwargs": {
                    "channels": [32, 64, 128, 256],
                }
            },
        },
        "mn-4x-s": {
            "inherits_from": "mn-4x",
            "patch_size_multiplier": 6,
            "architecture": {
                "arch_kwargs": {
                    "channels": [64, 128, 256, 512],
                }
            },
        },
        "mn-4x-m": {
            "inherits_from": "mn-4x",
            "patch_size_multiplier": 6,
            "architecture": {
                "arch_kwargs": {
                    "channels": [96, 192, 384, 768],
                }
            },
        },
    }


class TemporaryPlanner(BaselinePlanner):
    _presets = {
        "mn-4x-t_000": {
            "inherits_from": "mn-4x-t",
        }
    }
