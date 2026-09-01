"""Adapters for official BackdoorBench checkpoints and CIFAR-10 normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.247, 0.243, 0.261)


class NormalizedClassifier(nn.Module):
    """Apply BackdoorBench's CIFAR-10 normalization before classification."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, raw_images: torch.Tensor) -> torch.Tensor:
        """Classify raw [0, 1] images with the training-time normalization."""

        return self.model((raw_images - self.mean) / self.std)


def _backdoorbench_model_factory(backdoorbench_root: str):
    """Import the official factory without copying or reimplementing architectures."""

    root = str(Path(backdoorbench_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from utils.aggregate_block.model_trainer_generate import generate_cls_model

    return generate_cls_model


def load_attack_result_model(
    result_path: str,
    *,
    backdoorbench_root: str,
    device: torch.device,
) -> tuple[NormalizedClassifier, dict]:
    """Load a BackdoorBench attack_result.pt and return a normalized classifier."""

    result = torch.load(result_path, map_location="cpu", weights_only=False)
    required = {"model_name", "num_classes", "model"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"attack result is missing keys: {sorted(missing)}")

    factory = _backdoorbench_model_factory(backdoorbench_root)
    model = factory(result["model_name"], result["num_classes"], image_size=32)
    state = result["model"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    wrapped = NormalizedClassifier(model).to(device).eval()
    for parameter in wrapped.parameters():
        parameter.requires_grad_(False)
    return wrapped, result
