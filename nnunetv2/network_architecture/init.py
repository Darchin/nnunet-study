import torch
import torch.nn as nn


def zero_init(module: nn.Module):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.zeros_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)
        
def identity_init(module: nn.Module):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.ones_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)