from typing import List, Tuple, Union

from nnunetv2.experiment_planning.experiment_planners.stemmed_planner import (
    StemmedPlanner,
)

from itertools import product


class MobileUNetPlanner(StemmedPlanner):
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

    @property
    def configs(self):
        return {
            "MN": {
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
                    "architecture.arch_kwargs.ndim",
                    "architecture.arch_kwargs.kernel_sizes",
                    "architecture.arch_kwargs.strides",
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
            "MN-2x": {
                "inherits_from": ["2x", "MN"],
                "architecture": {
                    "arch_kwargs": {
                        "ndim": 3,
                        "kernel_sizes": [[3] * 3 for _ in range(5)],
                        "strides": [[1] * 3] + [[2] * 3 for _ in range(5 - 1)],
                        "encoder_depths": [2, 3, 3, 9, 3],
                        "decoder_depths": [1, 1, 1, 1],
                        "encoder_expansion_ratios": [1.0, 2.0, 3.0, 4.0, 4.0],
                        "decoder_expansion_ratios": [1.0, 2.0, 3.0, 4.0],
                    }
                },
            },
            "MN-3x": {
                "inherits_from": ["3x", "MN"],
                "architecture": {
                    "arch_kwargs": {
                        "ndim": 3,
                        "kernel_sizes": [[3] * 3 for _ in range(4)],
                        "strides": [[1] * 3] + [[2] * 3 for _ in range(4 - 1)],
                        "encoder_depths": [3, 3, 9, 3],
                        "decoder_depths": [1, 1, 1],
                        "encoder_expansion_ratios": [1.0, 2.0, 3.0, 4.0],
                        "decoder_expansion_ratios": [1.0, 2.0, 3.0],
                    }
                },
            },
            "MN-4x": {
                "inherits_from": ["4x", "MN"],
                "architecture": {
                    "arch_kwargs": {
                        "ndim": 3,
                        "kernel_sizes": [[3] * 3 for _ in range(4)],
                        "strides": [[1] * 3] + [[2] * 3 for _ in range(4 - 1)],
                        "encoder_depths": [3, 3, 9, 3],
                        "decoder_depths": [1, 1, 1],
                        "encoder_expansion_ratios": [2.0, 3.0, 4.0, 4.0],
                        "decoder_expansion_ratios": [2.0, 3.0, 4.0],
                    }
                },
            },
            "MN-2x-T": {
                "inherits_from": "MN-2x",
                "patch_size_multiplier": 6,
                "architecture": {"arch_kwargs": {"channels": [32, 64, 96, 160]}},
            },
            "MN-2x-S": {
                "inherits_from": "MN-2x",
                "patch_size_multiplier": 6,
                "architecture": {"arch_kwargs": {"channels": [32, 64, 128, 192, 320]}},
            },
            "MN-2x-M": {
                "inherits_from": "MN-2x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [48, 96, 192, 288, 480],
                    }
                },
            },
            "MN-3x-T": {
                "inherits_from": "MN-3x",
                "patch_size_multiplier": 8,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [32, 64, 96, 160],
                    }
                },
            },
            "MN-3x-S": {
                "inherits_from": "MN-3x",
                "patch_size_multiplier": 8,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [64, 128, 192, 320],
                    }
                },
            },
            "MN-3x-M": {
                "inherits_from": "MN-3x",
                "patch_size_multiplier": 8,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [96, 192, 288, 480],
                    }
                },
            },
            "MN-3x-L": {
                "inherits_from": "MN-3x",
                "patch_size_multiplier": 8,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [128, 256, 384, 640],
                    }
                },
            },
            "MN-4x-T": {
                "inherits_from": "MN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [32, 64, 96, 160],
                    }
                },
            },
            "MN-4x-S-IN+ReLU": {
                "inherits_from": "MN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "norm_layer": "nnunetv2.network_architecture.nd.InstanceNormNd",
                        "act_layer": "torch.nn.ReLU",
                        "channels": [64, 128, 192, 320],
                    }
                },
            },
            "MN-4x-M": {
                "inherits_from": "MN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [96, 192, 288, 480],
                    }
                },
            },
            "MN-4x-L": {
                "inherits_from": "MN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [128, 256, 384, 640],
                    }
                },
            },
        }

    def _additional_configurations(self) -> dict:
        return self.configs


class ConvNeXtPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        return {
            "CN": {
                "architecture": {
                    "network_class_name": "nnunetv2.network_architecture.mobile_unet.MobileUNet",
                    "arch_kwargs": {
                        "block_factory": "nnunetv2.network_architecture.uib.ConvNeXtBlock",
                        "norm_layer": "nnunetv2.network_architecture.nd.LayerNormNd",
                        "norm_kwargs": {},
                        "act_layer": "torch.nn.SiLU",
                        "act_kwargs": {"inplace": True},
                    },
                    "_kw_requires_import": ("block_factory", "norm_layer", "act_layer"),
                },
                "required_for_training": [
                    "architecture.arch_kwargs.ndim",
                    "architecture.arch_kwargs.kernel_sizes",
                    "architecture.arch_kwargs.strides",
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
            "CN-2x": {
                "inherits_from": ["2x", "CN"],
                "architecture": {
                    "arch_kwargs": {
                        "ndim": 3,
                        "kernel_sizes": [[3] * 3 for _ in range(5)],
                        "strides": [[1] * 3] + [[2] * 3 for _ in range(5 - 1)],
                        "encoder_depths": [2, 3, 3, 9, 3],
                        "decoder_depths": [1, 1, 1, 1],
                        "encoder_expansion_ratios": 4,
                        "decoder_expansion_ratios": 4,
                    }
                },
            },
            "CN-3x": {
                "inherits_from": ["3x", "CN"],
                "architecture": {
                    "arch_kwargs": {
                        "ndim": 3,
                        "kernel_sizes": [[3] * 3 for _ in range(4)],
                        "strides": [[1] * 3] + [[2] * 3 for _ in range(4 - 1)],
                        "encoder_depths": [3, 3, 9, 3],
                        "decoder_depths": [1, 1, 1],
                        "encoder_expansion_ratios": 4,
                        "decoder_expansion_ratios": 4,
                    }
                },
            },
            "CN-4x": {
                "inherits_from": ["4x", "CN"],
                "architecture": {
                    "arch_kwargs": {
                        "ndim": 3,
                        "kernel_sizes": [[3] * 3 for _ in range(4)],
                        "strides": [[1] * 3] + [[2] * 3 for _ in range(4 - 1)],
                        "encoder_depths": [3, 3, 9, 3],
                        "decoder_depths": [1, 1, 1],
                        "encoder_expansion_ratios": 4.0,
                        "decoder_expansion_ratios": 4.0,
                    }
                },
            },
            "CN-2x-T": {
                "inherits_from": "CN-2x",
                "patch_size_multiplier": 6,
                "architecture": {"arch_kwargs": {"channels": [32, 64, 96, 160]}},
            },
            "CN-2x-S": {
                "inherits_from": "CN-2x",
                "patch_size_multiplier": 6,
                "architecture": {"arch_kwargs": {"channels": [32, 64, 128, 192, 320]}},
            },
            "CN-2x-M": {
                "inherits_from": "CN-2x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [48, 96, 192, 288, 480],
                    }
                },
            },
            "CN-3x-T": {
                "inherits_from": "CN-3x",
                "patch_size_multiplier": 8,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [32, 64, 96, 160],
                    }
                },
            },
            "CN-3x-S": {
                "inherits_from": "CN-3x",
                "patch_size_multiplier": 8,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [64, 128, 192, 320],
                    }
                },
            },
            "CN-3x-M": {
                "inherits_from": "CN-3x",
                "patch_size_multiplier": 8,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [96, 192, 288, 480],
                    }
                },
            },
            "CN-3x-L": {
                "inherits_from": "CN-3x",
                "patch_size_multiplier": 8,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [128, 256, 384, 640],
                    }
                },
            },
            "CN-4x-T": {
                "inherits_from": "CN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [32, 64, 96, 160],
                    }
                },
            },
            "CN-4x-S": {
                "inherits_from": "CN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [64, 128, 192, 320],
                    }
                },
            },
            "CN-4x-M": {
                "inherits_from": "CN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [96, 192, 288, 480],
                    }
                },
            },
            "CN-4x-L": {
                "inherits_from": "CN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [128, 256, 384, 640],
                    }
                },
            },
        }


class SetOnePlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        norm_configs = {
            "IN": {
                "architecture": {
                    "arch_kwargs": {
                        "norm_layer": "nnunetv2.network_architecture.nd.InstanceNormNd"
                    }
                }
            },
            "LN": {
                "architecture": {
                    "arch_kwargs": {
                        "norm_layer": "nnunetv2.network_architecture.nd.LayerNormNd"
                    }
                }
            },
            "GN-G1": {
                "architecture": {
                    "arch_kwargs": {
                        "norm_layer": "nnunetv2.network_architecture.nd.GroupNormNd",
                        "norm_kwargs": {"num_groups": 1},
                    }
                }
            },
            "GN-CH16": {
                "architecture": {
                    "arch_kwargs": {
                        "norm_layer": "nnunetv2.network_architecture.nd.GroupNormNd",
                        "norm_kwargs": {"num_channels_per_group": 16},
                    }
                }
            },
        }
        block_configs = {
            "CN": {
                "architecture": {
                    "arch_kwargs": {
                        "block_factory": "nnunetv2.network_architecture.uib.ConvNeXtBlock"
                    }
                }
            },
            "IB": {
                "architecture": {
                    "arch_kwargs": {
                        "block_factory": "nnunetv2.network_architecture.uib.InvertedBottleneckBlock"
                    }
                }
            },
            "PMLP": {
                "architecture": {
                    "arch_kwargs": {
                        "block_factory": "nnunetv2.network_architecture.uib.PreDWMultilayerPerceptronBlock"
                    }
                }
            },
            "PIB": {
                "architecture": {
                    "arch_kwargs": {
                        "block_factory": "nnunetv2.network_architecture.uib.PreDWInvertedBottleneckBlock"
                    }
                }
            },
        }

        new_configs = configs | norm_configs | block_configs
        for block, norm in zip(block_configs, norm_configs):
            new_configs[f"MN-4x-S_{block}_{norm}"] = {"inherits_from": ["MN-4x-S", block, norm]}
        
        return new_configs


class MobileUNetMoEPlanner(MobileUNetPlanner):
    @property
    def moe_configs(self):
        # PARAMS = {
        #     "num_experts": [2, 4, 8, 16],
        #     "pw_backend": [None, "bmm"],
        #     "dw_backend": [None, "bg"],
        #     "router_kernel_size": [1, 3],
        #     "router_stride": [1, 2],
        #     "router_seq_op": [
        #         ["gap", "conv", "sigmoid", "norm"],
        #         ["conv", "sigmoid", "gap", "norm"],
        #     ],
        # }

        # alias = "MN-4x-S"
        # for k, v in PARAMS.items():
        #     alias += "-E{num_experts}"

        configs = {
            {
                "MN-4x-S_MoE-S1toS4-E4-PW+BMM-GAPConvSigNorm": {
                    "inherits_from": "MN-4x-S",
                    "architecture": {
                        "arch_kwargs": {
                            "moe_configs": [
                                {
                                    "num_experts": 4,
                                    "pw_backend": "bmm",
                                    "router_kernel_size": 1,
                                    "router_stride": 1,
                                    "router_op_seq": ["gap", "conv", "sigmoid", "norm"],
                                }
                            ]
                            * 4
                        }
                    },
                },
                "MN-4x-S_MoE-S2toS4-E4-PW+BMM-GAPConvSigNorm": {
                    "inherits_from": "MN-4x-S",
                    "architecture": {
                        "arch_kwargs": {
                            "moe_configs": [{}]
                            + [
                                {
                                    "num_experts": 4,
                                    "pw_backend": "bmm",
                                    "router_kernel_size": 1,
                                    "router_stride": 1,
                                    "router_op_seq": ["gap", "conv", "sigmoid", "norm"],
                                }
                            ]
                            * 3
                        }
                    },
                },
                "MN-4x-S_MoE-S2toS4-E4-PW+BMM-DW+BG-GAPConvSigNorm": {
                    "inherits_from": "MN-4x-S",
                    "architecture": {
                        "arch_kwargs": {
                            "moe_configs": [{}]
                            + [
                                {
                                    "num_experts": 4,
                                    "pw_backend": "bmm",
                                    "dw_backend": "bg",
                                    "router_kernel_size": 1,
                                    "router_stride": 1,
                                    "router_op_seq": ["gap", "conv", "sigmoid", "norm"],
                                }
                            ]
                            * 3
                        }
                    },
                },
                "MN-4x-S_MoE-S2toS4-E4-PW+BMM-K1S1-ConvSigGAPNorm": {
                    "inherits_from": "MN-4x-S",
                    "architecture": {
                        "arch_kwargs": {
                            "moe_configs": [{}]
                            + [
                                {
                                    "num_experts": 4,
                                    "pw_backend": "bmm",
                                    "router_kernel_size": 1,
                                    "router_stride": 1,
                                    "router_op_seq": ["conv", "sigmoid", "gap", "norm"],
                                }
                            ]
                            * 3
                        }
                    },
                },
            }
        }
        return configs

    def _additional_configurations(self):
        return self.configs | self.moe_configs
