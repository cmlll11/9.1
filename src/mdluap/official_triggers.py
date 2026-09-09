"""Official test-time trigger adapters used by the Stage 1D pilot.

The adapters intentionally fail closed for attack families whose test-time
state is unavailable.  They never replace a missing sample-specific or
input-dependent trigger with a hand-written patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class InputAwareGenerator(nn.Sequential):
    """The generator architecture used by BackdoorBench InputAware."""

    def __init__(self, input_channels: int = 3, output_channels: int | None = None):
        super().__init__()
        channel_init = 32
        steps = 3
        current = input_channels
        next_channels = channel_init
        for step in range(steps):
            self.add_module(f"convblock_down_{2 * step}", Conv2dBlock(current, next_channels))
            self.add_module(f"convblock_down_{2 * step + 1}", Conv2dBlock(next_channels, next_channels))
            self.add_module(f"downsample_{step}", nn.MaxPool2d(kernel_size=2, stride=2))
            if step < steps - 1:
                current = next_channels
                next_channels *= 2
        self.add_module("convblock_middle", Conv2dBlock(next_channels, next_channels))
        current = next_channels
        next_channels = current // 2
        for step in range(steps):
            self.add_module(f"upsample_{step}", nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.add_module(f"convblock_up_{2 * step}", Conv2dBlock(current, current))
            relu = step != steps - 1
            self.add_module(f"convblock_up_{2 * step + 1}", Conv2dBlock(current, next_channels, relu=relu))
            current = next_channels
            next_channels //= 2
            if step == steps - 2:
                next_channels = input_channels if output_channels is None else output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return torch.tanh(x) / (2.0 + 1e-7) + 0.5


class Conv2dBlock(nn.Module):
    """BackdoorBench InputAware convolution block."""

    def __init__(self, in_channels: int, out_channels: int, *, relu: bool = True):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05, affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x


class InputAwareThreshold(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x * 20.0 - 10.0) / (2.0 + 1e-7) + 0.5


@dataclass
class TriggerStatus:
    """Availability and provenance of one official test-time trigger."""

    trigger_type: str
    source: str | None
    available: bool
    reason: str | None = None


class TriggerAdapter:
    """Common interface for raw ``[0,1]`` test-time trigger transforms."""

    def __init__(self, status: TriggerStatus):
        self.status = status

    def apply(
        self,
        images: torch.Tensor,
        *,
        sample_indices: list[int],
        split: str,
    ) -> torch.Tensor:
        raise NotImplementedError


class UnavailableTrigger(TriggerAdapter):
    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        raise RuntimeError(self.status.reason or f"trigger unavailable: {self.status.trigger_type}")


class FixedImageTrigger(TriggerAdapter):
    def __init__(self, status: TriggerStatus, trigger: torch.Tensor, *, kind: str, alpha: float = 0.2):
        super().__init__(status)
        self.trigger = trigger
        self.kind = kind
        self.alpha = float(alpha)

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        trigger = self.trigger.to(device=images.device, dtype=images.dtype).unsqueeze(0)
        if self.kind == "badnet":
            return torch.where(trigger > 0, trigger, images)
        if self.kind == "blended" or self.kind == "adaptive_blend":
            return ((1.0 - self.alpha) * images + self.alpha * trigger).clamp(0.0, 1.0)
        raise ValueError(f"unsupported fixed trigger kind: {self.kind}")


class WaNetTrigger(TriggerAdapter):
    def __init__(self, status: TriggerStatus, identity_grid: torch.Tensor, noise_grid: torch.Tensor, *, s: float, grid_rescale: float):
        super().__init__(status)
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid
        self.s = float(s)
        self.grid_rescale = float(grid_rescale)

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        grid = (self.identity_grid + self.s * self.noise_grid / images.shape[-2]) * self.grid_rescale
        grid = torch.clamp(grid, -1.0, 1.0).expand(images.shape[0], -1, -1, -1)
        return F.grid_sample(images, grid, align_corners=True)


class InputAwareTrigger(TriggerAdapter):
    def __init__(self, status: TriggerStatus, generator: nn.Module, mask: nn.Module, threshold: nn.Module):
        super().__init__(status)
        self.generator = generator.eval()
        self.mask = mask.eval()
        self.threshold = threshold.eval()

    @torch.no_grad()
    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        pattern = self.generator(images)
        mask = self.threshold(self.mask(images))
        return (images + (pattern - images) * mask).clamp(0.0, 1.0)


class SSBAArrayTrigger(TriggerAdapter):
    def __init__(self, status: TriggerStatus, test_images: torch.Tensor):
        super().__init__(status)
        self.test_images = test_images

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        if split != "cifar100_test":
            raise ValueError("Stage 1D SSBA uses the CIFAR-100 test split")
        if not sample_indices or max(sample_indices) >= len(self.test_images):
            raise IndexError(
                f"SSBA test array has {len(self.test_images)} rows but requested index {max(sample_indices)}"
            )
        return self.test_images[torch.as_tensor(sample_indices, dtype=torch.long)].to(images.device, images.dtype)


def _load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def _load_array(path: Path) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(np.load(path))).float()
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, 32, 32):
        raise ValueError(f"SSBA array must have shape [N,3,32,32], got {tuple(tensor.shape)} from {path}")
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).contiguous()


def _load_wanet(path: Path, device: torch.device, *, s: float, grid_rescale: float) -> TriggerAdapter:
    state = torch.load(path, map_location=device, weights_only=False)
    identity = state["identity_grid"].to(device=device, dtype=torch.float32)
    noise = state["noise_grid"].to(device=device, dtype=torch.float32)
    status = TriggerStatus("wanet", str(path), True)
    return WaNetTrigger(status, identity, noise, s=s, grid_rescale=grid_rescale)


def _find_first(root: Path, names: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _group_seed_root(model_root: Path, group: str) -> Path:
    """Resolve common artifact-directory spellings used by server runs."""

    aliases = {
        "inputaware": ("inputaware", "input_aware", "input-aware"),
        "adaptive_blend": ("adaptive_blend", "adaptive-blend", "adaptiveblend", "adap_blend"),
    }
    for candidate in aliases.get(group, (group,)):
        path = model_root / candidate / "seed0"
        if path.exists():
            return path
    return model_root / group / "seed0"


def _load_inputaware(path: Path, device: torch.device) -> TriggerAdapter:
    artifact: Any = torch.load(path, map_location=device, weights_only=False)
    if "best_trigger" in artifact:
        artifact = artifact["best_trigger"]
    generator = InputAwareGenerator().to(device)
    mask = InputAwareGenerator(output_channels=1).to(device)
    generator.load_state_dict(artifact["generator"])
    mask.load_state_dict(artifact["mask"])
    status = TriggerStatus("inputaware", str(path), True)
    return InputAwareTrigger(status, generator, mask, InputAwareThreshold().to(device))


def build_trigger_adapters(
    *,
    backdoorbench_root: Path,
    model_root: Path,
    record_root: Path | None,
    device: torch.device,
    wanet_s: float,
    wanet_grid_rescale: float,
    blended_alpha: float,
    adaptive_blend_alpha: float,
    explicit: dict[str, Path | None],
) -> dict[str, TriggerAdapter]:
    """Build official test-time adapters without synthetic fallbacks."""

    adapters: dict[str, TriggerAdapter] = {}
    badnet_path = explicit.get("badnet") or backdoorbench_root / "resource" / "badnet" / "trigger_image.png"
    blended_path = explicit.get("blended") or backdoorbench_root / "resource" / "blended" / "hello_kitty.jpeg"
    adaptive_path = explicit.get("adaptive_blend")
    lf_path = backdoorbench_root / "resource" / "lowFrequency" / "cifar10_preactresnet18_0_255.npy"

    if badnet_path and badnet_path.is_file():
        adapters["badnet"] = FixedImageTrigger(TriggerStatus("badnet", str(badnet_path), True), _load_image(badnet_path), kind="badnet")
    else:
        adapters["badnet"] = UnavailableTrigger(TriggerStatus("badnet", str(badnet_path), False, "official BadNet trigger missing"))

    if blended_path and blended_path.is_file():
        adapters["blended"] = FixedImageTrigger(TriggerStatus("blended", str(blended_path), True), _load_image(blended_path), kind="blended", alpha=blended_alpha)
    else:
        adapters["blended"] = UnavailableTrigger(TriggerStatus("blended", str(blended_path), False, "official Blended test trigger missing"))

    wanet_path = explicit.get("wanet") or model_root / "wanet" / "seed0" / "state_dict.pt"
    if not wanet_path.is_file() and record_root is not None:
        wanet_path = _find_first(record_root, ("mdl_uap_hard_wanet_seed0/state_dict.pt", "mdl_uap_wanet_seed0/state_dict.pt")) or wanet_path
    if wanet_path and wanet_path.is_file():
        try:
            adapters["wanet"] = _load_wanet(wanet_path, device, s=wanet_s, grid_rescale=wanet_grid_rescale)
        except Exception as exc:
            adapters["wanet"] = UnavailableTrigger(TriggerStatus("wanet", str(wanet_path), False, str(exc)))
    else:
        adapters["wanet"] = UnavailableTrigger(TriggerStatus("wanet", str(wanet_path), False, "WaNet test-time grid state missing"))

    ssba_path = explicit.get("ssba")
    if ssba_path is None:
        ssba_path = _find_first(_group_seed_root(model_root, "ssba"), ("test.npy", "ssba_test.npy", "cifar100_ssba_test.npy"))
    if ssba_path is None and record_root is not None:
        ssba_path = _find_first(record_root, (
            "mdl_uap_hard_ssba_seed0/cifar100_ssba_test.npy",
            "mdl_uap_hard_ssba_seed0/ssba_test.npy",
            "mdl_uap_hard_ssba_seed0/test_replace_imgs.npy",
        ))
    if ssba_path and ssba_path.is_file():
        try:
            adapters["ssba"] = SSBAArrayTrigger(TriggerStatus("ssba", str(ssba_path), True), _load_array(ssba_path))
        except Exception as exc:
            adapters["ssba"] = UnavailableTrigger(TriggerStatus("ssba", str(ssba_path), False, str(exc)))
    else:
        adapters["ssba"] = UnavailableTrigger(TriggerStatus("ssba", str(ssba_path), False, "official SSBA CIFAR-100 test-time array missing"))

    inputaware_path = explicit.get("inputaware") or _find_first(_group_seed_root(model_root, "inputaware"), ("trigger_state.pt", "inputaware_trigger_state.pt"))
    if inputaware_path is None and record_root is not None:
        inputaware_path = _find_first(record_root, (
            "mdl_uap_hard_inputaware_seed0/trigger_state.pt",
            "mdl_uap_inputaware_seed0/trigger_state.pt",
        ))
    if inputaware_path and inputaware_path.is_file():
        try:
            adapters["inputaware"] = _load_inputaware(inputaware_path, device)
        except Exception as exc:
            adapters["inputaware"] = UnavailableTrigger(TriggerStatus("inputaware", str(inputaware_path), False, str(exc)))
    else:
        adapters["inputaware"] = UnavailableTrigger(TriggerStatus("inputaware", str(inputaware_path), False, "Input-Aware generator/mask state missing"))

    if adaptive_path is None:
        adaptive_path = _find_first(_group_seed_root(model_root, "adaptive_blend"), ("trigger.png", "trigger.jpg", "adaptive_blend_trigger.png"))
    if adaptive_path is None and record_root is not None:
        adaptive_path = _find_first(record_root, (
            "mdl_uap_hard_adaptive_blend_seed0/adaptive_blend_trigger.png",
            "mdl_uap_adaptive_blend_seed0/adaptive_blend_trigger.png",
        ))
    if adaptive_path and adaptive_path.is_file():
        adapters["adaptive_blend"] = FixedImageTrigger(
            TriggerStatus("adaptive_blend", str(adaptive_path), True),
            _load_image(adaptive_path), kind="adaptive_blend", alpha=adaptive_blend_alpha,
        )
    else:
        adapters["adaptive_blend"] = UnavailableTrigger(TriggerStatus("adaptive_blend", str(adaptive_path), False, "official Adaptive-Blend test trigger missing"))

    return adapters
