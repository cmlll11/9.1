"""Adapters for official BackdoorBench checkpoints and CIFAR-10 normalization."""

from __future__ import annotations

import sys
import importlib.util
import os
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


def load_modelzoo_classifier(
    alias: str,
    *,
    model_zoo_root: str,
    device: torch.device,
) -> tuple[NormalizedClassifier, dict]:
    """Load one registered model through the public Model Zoo API.

    Model Zoo checkpoints are stored as models that consume normalized CIFAR-10
    tensors.  The experiment itself uses raw ``[0, 1]`` image tensors, so the
    normalization boundary is kept here rather than duplicated by callers.
    """

    try:
        from modelzoo import get_model_info, load_model
    except ImportError as exc:  # pragma: no cover - exercised on the server
        raise ImportError(
            "The shared Model Zoo package is required. Install it with "
            "pip install -e /path/to/backdoor-model-zoo and set MODEL_ZOO_ROOT."
        ) from exc

    # Model Zoo 0.1.0 resolves registry metadata from MODEL_ZOO_ROOT and
    # exposes get_model_info(alias) without a root keyword.  Keep a fallback
    # for newer releases that accept root explicitly.
    os.environ["MODEL_ZOO_ROOT"] = str(Path(model_zoo_root).expanduser())
    try:
        info = get_model_info(alias)
    except TypeError:
        info = get_model_info(alias, root=model_zoo_root)
    model = load_model(alias, device=device, root=model_zoo_root)
    wrapped = NormalizedClassifier(model).to(device).eval()
    for parameter in wrapped.parameters():
        parameter.requires_grad_(False)
    return wrapped, info


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


def load_clean_model(
    result_path: str,
    *,
    backdoorbench_root: str,
    device: torch.device,
) -> tuple[NormalizedClassifier, dict]:
    """Load the official BackdoorBench ``clean_model.pth`` state dictionary."""

    state = torch.load(result_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Clean checkpoint is not a state dictionary: {result_path}")

    factory = _backdoorbench_model_factory(backdoorbench_root)
    model = factory("preactresnet18", 10, image_size=32)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    wrapped = NormalizedClassifier(model).to(device).eval()
    for parameter in wrapped.parameters():
        parameter.requires_grad_(False)
    return wrapped, {
        "model_name": "preactresnet18",
        "num_classes": 10,
        "checkpoint_format": "clean_model.pth",
        "checkpoint_path": str(Path(result_path).resolve()),
    }


def load_model_checkpoint(
    result_path: str,
    *,
    backdoorbench_root: str,
    device: torch.device,
) -> tuple[NormalizedClassifier, dict]:
    """Load either an official Clean or BackdoorBench attack checkpoint."""

    if Path(result_path).name == "clean_model.pth":
        return load_clean_model(
            result_path,
            backdoorbench_root=backdoorbench_root,
            device=device,
        )
    return load_attack_result_model(
        result_path,
        backdoorbench_root=backdoorbench_root,
        device=device,
    )


def load_backdoor_toolbox_resnet18(
    model_path: str,
    *,
    backdoor_toolbox_root: str,
    device: torch.device,
) -> tuple[NormalizedClassifier, dict]:
    """Load the official backdoor-toolbox CIFAR-10 ResNet-18 checkpoint.

    Adaptive-Blend is not serialized as a BackdoorBench ``attack_result.pt``.
    The official toolbox stores the classifier state dictionary directly, so
    this adapter imports the toolbox's own ``utils/resnet.py`` architecture
    and applies the same CIFAR-10 normalization used by the toolbox.
    """

    root = Path(backdoor_toolbox_root).resolve()
    source = root / "utils" / "resnet.py"
    if not source.is_file():
        raise FileNotFoundError(f"backdoor-toolbox ResNet source not found: {source}")
    module_name = "stage1d_backdoor_toolbox_resnet"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import backdoor-toolbox architecture from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Adaptive-Blend checkpoint is not a state dictionary: {model_path}")
    state = {key.removeprefix("module."): value for key, value in payload.items()}
    model = module.ResNet18(num_classes=10)
    model.load_state_dict(state, strict=True)
    wrapped = NormalizedClassifier(model).to(device).eval()
    for parameter in wrapped.parameters():
        parameter.requires_grad_(False)
    return wrapped, {
        "model_name": "backdoor_toolbox.ResNet18",
        "num_classes": 10,
        "model_path": str(Path(model_path).resolve()),
        "source": "official backdoor-toolbox",
    }
