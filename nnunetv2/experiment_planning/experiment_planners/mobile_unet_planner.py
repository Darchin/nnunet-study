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
                    "num_epochs": 1000,
                    "warmup_epochs": 5,
                    "min_lr": 1e-6,
                    "enable_deep_supervision": False,
                    "2d_aug": None,
                    "use_nn_seg_resample": False,
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
                        "encoder_expansion_ratios": [2.0, 3.0, 4.0, 4.0],
                        "decoder_expansion_ratios": [2.0, 3.0, 4.0],
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
            "MN-2x+2x": {
                "inherits_from": ["2x+2x", "MN"],
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
            "MN-1.5x+2x": {
                "inherits_from": ["1.5x+2x", "MN"],
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
            "MN-2x+2x-hires": {
                "inherits_from": ["2x+2x-hires", "MN"],
                "architecture": {
                    "arch_kwargs": {
                        "ndim": 3,
                        "kernel_sizes": [[3] * 3 for _ in range(4)],
                        "strides": [[1] * 3] + [[2] * 3 for _ in range(3)],
                        "encoder_depths": [3, 3, 9, 3],
                        "decoder_depths": [1, 1, 1],
                        "encoder_expansion_ratios": [2.0, 3.0, 4.0, 4.0],
                        "decoder_expansion_ratios": [2.0, 3.0, 4.0],
                    }
                },
            },
            "MN-1.5x+2x-hires": {
                "inherits_from": ["1.5x+2x-hires", "MN"],
                "architecture": {
                    "arch_kwargs": {
                        "ndim": 3,
                        "kernel_sizes": [[3] * 3 for _ in range(4)],
                        "strides": [[1] * 3] + [[2] * 3 for _ in range(3)],
                        "encoder_depths": [3, 3, 9, 3],
                        "decoder_depths": [1, 1, 1],
                        "encoder_expansion_ratios": [2.0, 3.0, 4.0, 4.0],
                        "decoder_expansion_ratios": [2.0, 3.0, 4.0],
                    }
                },
            },
            "MN-2x-S": {
                "inherits_from": "MN-2x",
                "patch_size_multiplier": 4,
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
            "MN-4x-S": {
                "inherits_from": "MN-4x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [64, 128, 192, 320],
                    }
                },
            },
            "MN-2x+2x-S": {
                "inherits_from": "MN-2x+2x",
                "patch_size_multiplier": 6,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [64, 128, 192, 320],
                    }
                },
            },
            "MN-1.5x+2x-S": {
                "inherits_from": "MN-1.5x+2x",
                "patch_size_multiplier": 5,
                "architecture": {
                    "arch_kwargs": {
                        "channels": [64, 128, 192, 320],
                    }
                },
            },
            "MN-2x+2x-hires-S": {
                "inherits_from": "MN-2x+2x-hires",
                "patch_size_multiplier": 6,
                "architecture": {"arch_kwargs": {"channels": [64, 128, 192, 320]}},
            },
            "MN-1.5x+2x-hires-S": {
                "inherits_from": "MN-1.5x+2x-hires",
                "patch_size_multiplier": 5,
                "architecture": {"arch_kwargs": {"channels": [64, 128, 192, 320]}},
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
            "MN-2x+2x-M": {
                "inherits_from": "MN-2x+2x",
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


class BlockDesignPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        norm = {
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
            "GN-C16": {
                "architecture": {
                    "arch_kwargs": {
                        "norm_layer": "nnunetv2.network_architecture.nd.GroupNormNd",
                        "norm_kwargs": {"num_channels_per_group": 16},
                    }
                }
            },
        }

        block = {
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

        scale = ("S", "M", "L")

        new_configs = configs | norm | block
        for b, n, s in product(block, norm, scale):
            new_configs[f"MN-4x-{s}_{b}_{n}"] = {
                "inherits_from": [f"MN-4x-{s}", b, n],
                "patch_size_multiplier": 6,
                "trainer": {"num_epochs": 500},
            }

        return new_configs


class PatchSizeAndTrainingDurationPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        psm_to_duration_mapping = {
            4: [844, 1688, 3375],
            6: [250, 500, 1000],
        }

        new_configs = configs
        for psm in psm_to_duration_mapping.keys():
            for ne in psm_to_duration_mapping[psm]:
                for s in ("S", "M"):
                    new_configs[f"MN-4x-{s}_P{psm}_EP{ne}"] = {
                        "inherits_from": [f"MN-4x-{s}"],
                        "patch_size_multiplier": psm,
                        "trainer": {"num_epochs": ne},
                    }

        return new_configs


class SEPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        placements = {"Enc1+": "encoder", "Dec1+": "decoder"}

        new_configs = configs
        for alias, placement in placements.items():
            new_configs[f"MN-4x-S_SE-{alias}"] = {
                "inherits_from": [f"MN-4x-S"],
                "architecture": {
                    "arch_kwargs": {
                        f"{placement}_se_configs": {
                            "reduction": 4.0,
                            "placement": "mid",
                        }
                    }
                },
            }

        return new_configs


class MoERouterActPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        router_op_seqs = {
            "Softmax": ["conv", "softmax", "gap"],
            "SigNormGAP": ["conv", "sigmoid", "norm", "gap"],
            "SigGAPNorm": ["conv", "sigmoid", "gap", "norm"],
            "SpRootNormGAP": ["conv", "sproot", "norm", "gap"],
            "SpRootGAPNorm": ["conv", "sproot", "gap", "norm"],
        }

        new_configs = configs
        for alias, ros in router_op_seqs.items():
            new_configs[f"MN-4x-S_MoE-{alias}-K3-S2-PW+DW-Enc1+-E4"] = {
                "inherits_from": [f"MN-4x-S"],
                "architecture": {
                    "arch_kwargs": {
                        f"encoder_moe_configs": {
                            "num_experts": 4,
                            "pw_backend": "bmm",
                            "dw_backend": "bag",
                            "router_kernel_size": 3,
                            "router_stride": 2,
                            "router_op_seq": ros,
                        }
                    }
                },
                "trainer": {
                    "router_schedule": (
                        {"start_val": 30.0, "end_val": 1.0, "end_epoch": 25}
                        if alias == "Softmax"
                        else None
                    ),
                },
            }

        return new_configs


SELECTED_ROUTER_ACT = {
    "alias": "SigGAPNorm",
    "router_op_seq": ["conv", "sigmoid", "gap", "norm"],
}


class MoEStagePlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        cases = {
            "Enc1+": [True] * 4,
            "Enc2+": [False] + [True] * 3,
            "Dec1+": [True] * 3,
            "Dec2+": [False] + [True] * 2,
        }

        moe_config = {
            "num_experts": 4,
            "pw_backend": "bmm",
            "dw_backend": "bag",
            "router_kernel_size": 3,
            "router_stride": 2,
            "router_op_seq": SELECTED_ROUTER_ACT["router_op_seq"],
        }

        new_configs = configs
        for alias, c in cases.items():
            xcoder = "encoder" if alias.startswith("Enc") else "decoder"
            new_configs[
                f"MN-4x-S_MoE-{SELECTED_ROUTER_ACT["alias"]}-K3-S2-PW+DW-{alias}-E4"
            ] = {
                "inherits_from": [f"MN-4x-S"],
                "architecture": {
                    "arch_kwargs": {
                        f"{xcoder}_moe_configs": [
                            {} if c[i] == False else moe_config for i in range(len(c))
                        ]
                    }
                },
            }

        return new_configs


SELECTED_STAGES = {
    "alias": "Enc2+",
    "xcoder": "encoder",
    "stages": [False] + [True] * 3,
}


class MoELayerPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        cases = {
            "PW": {"pw_backend": "bmm"},
            "PW+DW": {"pw_backend": "bmm", "dw_backend": "bag"},
        }

        new_configs = configs
        for alias, c in cases.items():
            new_configs[
                f"MN-4x-S_MoE-{SELECTED_ROUTER_ACT["alias"]}-K3-S2-{alias}-{SELECTED_STAGES['alias']}-E4"
            ] = {
                "inherits_from": [f"MN-4x-S"],
                "architecture": {
                    "arch_kwargs": {
                        f"{SELECTED_STAGES["xcoder"]}_moe_configs": [
                            (
                                {}
                                if SELECTED_STAGES["stages"][i] == False
                                else {
                                    "num_experts": 4,
                                    **c,
                                    "router_kernel_size": 3,
                                    "router_stride": 2,
                                    "router_op_seq": SELECTED_ROUTER_ACT[
                                        "router_op_seq"
                                    ],
                                }
                            )
                            for i in range(len(SELECTED_STAGES["stages"]))
                        ]
                    }
                },
            }

        return new_configs


SELECTED_LAYERS = {"alias": "PW+DW", "pw_backend": "bmm", "dw_backend": "bag"}


class MoERouterConvPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        kernels = [1, 3]
        strides = [1, 2]

        new_configs = configs
        for k, s in product(kernels, strides):
            new_configs[
                f"MN-4x-S_MoE-{SELECTED_ROUTER_ACT["alias"]}-K{k}-S{s}-{SELECTED_LAYERS["alias"]}-{SELECTED_STAGES["alias"]}-E4"
            ] = {
                "inherits_from": [f"MN-4x-S"],
                "architecture": {
                    "arch_kwargs": {
                        f"encoder_moe_configs": [
                            (
                                {}
                                if SELECTED_STAGES["stages"][i] == False
                                else {
                                    "num_experts": 4,
                                    "pw_backend": SELECTED_LAYERS["pw_backend"],
                                    "dw_backend": SELECTED_LAYERS["dw_backend"],
                                    "router_kernel_size": k,
                                    "router_stride": s,
                                    "router_op_seq": SELECTED_ROUTER_ACT[
                                        "router_op_seq"
                                    ],
                                }
                            )
                            for i in range(len(SELECTED_STAGES["stages"]))
                        ]
                    }
                },
            }

        return new_configs


SELECTED_ROUTER_CONV = {"kernel": 3, "stride": 2}


class MoENumExpertsPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs

        num_experts = [2, 4, 8, 16]

        new_configs = configs
        for ne in num_experts:
            new_configs[
                f"MN-4x-S_MoE-{SELECTED_ROUTER_ACT["alias"]}-K{SELECTED_ROUTER_CONV["kernel"]}-S{SELECTED_ROUTER_CONV["stride"]}-{SELECTED_LAYERS["alias"]}-{SELECTED_STAGES["alias"]}-E{ne}"
            ] = {
                "inherits_from": [f"MN-4x-S"],
                "architecture": {
                    "arch_kwargs": {
                        f"encoder_moe_configs": [
                            (
                                {}
                                if SELECTED_STAGES["stages"][i] == False
                                else {
                                    "num_experts": ne,
                                    "pw_backend": SELECTED_LAYERS["pw_backend"],
                                    "dw_backend": SELECTED_LAYERS["dw_backend"],
                                    "router_kernel_size": SELECTED_ROUTER_CONV[
                                        "kernel"
                                    ],
                                    "router_stride": SELECTED_ROUTER_CONV["stride"],
                                    "router_op_seq": SELECTED_ROUTER_ACT[
                                        "router_op_seq"
                                    ],
                                }
                            )
                            for i in range(len(SELECTED_STAGES["stages"]))
                        ]
                    }
                },
            }

        return new_configs


class BraTS2024GLIPlanner(MobileUNetPlanner):
    @property
    def configs(self):
        configs = super().configs
        new_configs = configs

        new_configs["MN-1.5x+2x-hires-S_Static"] = {
            "inherits_from": ["MN-1.5x+2x-hires-S"],
            "patch_size_multiplier": 5,
            "trainer": {"num_epochs": 1000},
        }

        # new_configs["MN-4x-S_Dynamic"] = {
        #     "inherits_from": ["MN-4x-S_Static"],
        #     "architecture": {
        #         "arch_kwargs": {
        #             f"encoder_moe_configs": [
        #                 (
        #                     {}
        #                     if not is_moe_stage
        #                     else {
        #                         "num_experts": 4,
        #                         "pw_backend": "bmm",
        #                         "dw_backend": "bag",
        #                         "router_kernel_size": 3,
        #                         "router_stride": 2,
        #                         "router_op_seq": ["conv", "sigmoid", "gap", "norm"],
        #                     }
        #                 )
        #                 for is_moe_stage in [False, True, True, False]
        #             ]
        #         }
        #     },
        # }

        return new_configs
