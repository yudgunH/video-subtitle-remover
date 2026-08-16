"""Shared PyTorch inference contexts used by CUDA inpainting models."""

import os
from contextlib import nullcontext

import torch


def cuda_autocast(device):
    """Use FP16 autocast on CUDA, with an environment escape hatch."""
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    disabled = os.environ.get("VSR_DISABLE_FP16", "").lower() in {"1", "true", "yes"}
    if device_type == "cuda" and not disabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
