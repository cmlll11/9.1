"""Strict test-time trigger adapters for Stage 1D-WT.

The mechanism experiment must compare each model with the trigger that was
used by that attack.  This module therefore fails closed: a missing or
incompatible trigger state produces an unavailable adapter instead of a
synthetic replacement trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import importlib.util
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


@dataclass
class TriggerStatus:
    """Availability and provenance of an attack's test-time trigger."""

    trigger_type: str
    source: str | None
    available: bool
    reason: str | None = None


class TriggerAdapter:
    """Interface for applying a raw-pixel trigger to a batch."""

    def __init__(self, status: TriggerStatus):
        self.status = status

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        raise NotImplementedError


def _quantize_like_torchvision_save_image(values: torch.Tensor) -> torch.Tensor:
    """Quantize ``[0, 1]`` images exactly as torchvision ``save_image``.

    BackdoorBench's official SSBA pipeline calls ``torchvision.utils.save_image``
    on each encoder output.  The torchvision implementation converts floating
    point pixels with ``floor(255 * value + 0.5)`` when it casts to ``uint8``;
    this is *not* identical to ``torch.round`` at half-integer values because
    ``torch.round`` uses round-to-even semantics.  Keeping the quantization in
    one helper makes the in-memory SSBA trigger and the provenance checker use
    the same pixel convention as the official PNG files.

    Args:
        values: Image tensor with arbitrary leading dimensions and values
            expected in ``[0, 1]``.

    Returns:
        A float tensor whose values are the exact uint8 levels divided by 255.
        The shape and device match ``values``.
    """

    return torch.floor(values * 255.0 + 0.5).clamp(0.0, 255.0) / 255.0


class UnavailableTrigger(TriggerAdapter):
    """Adapter used when the official trigger cannot be reproduced."""

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        raise RuntimeError(self.status.reason or "trigger unavailable")


class FixedImageTrigger(TriggerAdapter):
    """BadNet patch or Blended fixed-image trigger."""

    def __init__(self, status: TriggerStatus, image: torch.Tensor, *, mode: str, alpha: float):
        super().__init__(status)
        self.image = image
        self.mode = mode
        self.alpha = float(alpha)

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        trigger = self.image.to(images.device, images.dtype).unsqueeze(0)
        if self.mode == "badnet":
            return torch.where(trigger > 0, trigger, images)
        if self.mode == "blended":
            # BackdoorBench's official test transform applies
            # np.clip(...).astype(np.uint8) after blending.  Reproduce that
            # floor-to-uint8 conversion before the Model Zoo normalization.
            blended = (
                (1.0 - self.alpha) * images.to(torch.float64)
                + self.alpha * trigger.to(torch.float64)
            ).clamp(0.0, 1.0)
            return (torch.floor(blended * 255.0).clamp(0.0, 255.0) / 255.0).to(images.dtype)
        return ((1.0 - self.alpha) * images + self.alpha * trigger).clamp(0.0, 1.0)


class WaNetTrigger(TriggerAdapter):
    """Official WaNet identity/noise-grid transformation."""

    def __init__(self, status: TriggerStatus, identity: torch.Tensor, noise: torch.Tensor, *, s: float, grid_rescale: float):
        super().__init__(status)
        self.identity = identity
        self.noise = noise
        self.s = float(s)
        self.grid_rescale = float(grid_rescale)

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        grid = (self.identity + self.s * self.noise / images.shape[-2]) * self.grid_rescale
        grid = grid.clamp(-1.0, 1.0).expand(images.shape[0], -1, -1, -1)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class SSBAArrayTrigger(TriggerAdapter):
    """Official SSBA sample-specific replacement images."""

    def __init__(self, status: TriggerStatus, replacements: torch.Tensor):
        super().__init__(status)
        self.replacements = replacements

    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        if split != "cifar100_test":
            raise ValueError("Stage 1D-WT SSBA expects a CIFAR-100 test replacement array")
        indices = torch.as_tensor(sample_indices, dtype=torch.long)
        if indices.numel() == 0 or int(indices.max()) >= len(self.replacements):
            raise IndexError("SSBA replacement array does not cover the requested dataset indices")
        return self.replacements[indices].to(images.device, images.dtype)


class InputAwareTrigger(TriggerAdapter):
    """Official BackdoorBench Input-Aware generator/mask composition."""

    def __init__(self, status: TriggerStatus, generator: nn.Module, mask: nn.Module, threshold: nn.Module):
        super().__init__(status)
        self.generator = generator.eval()
        self.mask = mask.eval()
        self.threshold = threshold.eval()
        self.mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        self.std = torch.tensor((0.247, 0.243, 0.261)).view(1, 3, 1, 1)

    @torch.no_grad()
    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        mean = self.mean.to(images.device, images.dtype)
        std = self.std.to(images.device, images.dtype)
        normalized = (images - mean) / std
        pattern = (self.generator(normalized) - mean) / std
        mask = self.threshold(self.mask(normalized))
        triggered_normalized = normalized + (pattern - normalized) * mask
        return (triggered_normalized * std + mean).clamp(0.0, 1.0)


class SSBAEncoderTrigger(TriggerAdapter):
    """Generate SSBA images with the exact trained StegaStamp encoder.

    The BackdoorBench SSBA attack stores pre-generated CIFAR-10 replacement
    arrays and does not expose an input-time transform.  This adapter is only
    enabled when a provenance JSON describes the encoder and fingerprint
    generation settings used to create that array.  It never indexes a
    CIFAR-10 replacement array for CIFAR-100 samples.
    """

    def __init__(
        self,
        status: TriggerStatus,
        encoder: nn.Module,
        provenance: dict[str, Any],
        *,
        bdb_root: Path,
        device: torch.device,
    ):
        super().__init__(status)
        self.encoder = encoder.eval()
        self.provenance = provenance
        self.bdb_root = bdb_root
        self.device = device
        self._fingerprints = self._build_fingerprints()

    def _build_fingerprints(self) -> torch.Tensor:
        values = self.provenance.get("fingerprint_values")
        length = int(self.provenance["fingerprint_length"])
        if values is not None:
            tensor = torch.as_tensor(values, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] != length:
                raise ValueError("SSBA fingerprint_values length does not match fingerprint_length")
            return tensor.to(self.device)

        method = str(self.provenance.get("encode_method", "bch")).lower()
        if method == "bch":
            module_path = self.bdb_root / "resource" / "ssba" / "generate_fingerprints.py"
            spec = importlib.util.spec_from_file_location("stage1d_ssba_fingerprints", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import SSBA fingerprint generator from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            secret = str(self.provenance["secret"])
            values = module.generate_fingerprints_from_bch(length, secret)
            return torch.as_tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)

        if method == "seed":
            seed = int(self.provenance.get("seed", 0))
            batch_size = int(self.provenance.get("batch_size", 32))
            generator = torch.Generator(device="cpu").manual_seed(seed)
            values = torch.randint(0, 2, (batch_size, length), generator=generator, dtype=torch.float32)
            if bool(self.provenance.get("identical_fingerprints", True)):
                values = values[:1]
            return values.to(self.device)

        raise ValueError(
            "SSBA provenance must provide fingerprint_values or a supported "
            f"encode_method (bch/seed), got {method!r}"
        )

    def _fingerprints_for(self, sample_indices: list[int]) -> torch.Tensor:
        if len(self._fingerprints) == 1:
            return self._fingerprints.expand(len(sample_indices), -1)
        batch_size = int(self.provenance.get("batch_size", len(self._fingerprints)))
        positions = torch.as_tensor([int(index) % batch_size for index in sample_indices], device=self.device)
        return self._fingerprints[positions]

    @torch.no_grad()
    def apply(self, images: torch.Tensor, *, sample_indices: list[int], split: str) -> torch.Tensor:
        if split != "cifar100_test":
            raise ValueError("SSBA encoder trigger is configured for CIFAR-100 test samples")
        fingerprints = self._fingerprints_for(sample_indices).to(images.device, images.dtype)
        output = self.encoder(fingerprints, images)
        if int(self.provenance.get("use_residual", 0)):
            output = images + output
        output = output.clamp(0.0, 1.0)
        if bool(self.provenance.get("quantize_uint8", True)):
            # The official embed_fingerprints.py writes every output with
            # torchvision.save_image().  Reproduce its half-up uint8
            # conversion before returning the trigger image to the experiment.
            output = _quantize_like_torchvision_save_image(output)
        return output


def _image(path: Path) -> torch.Tensor:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        # BackdoorBench decodes the official Blended JPEG with imageio before
        # converting it to a PIL image for the official resize path.
        import imageio.v2 as imageio

        image = Image.fromarray(np.asarray(imageio.imread(path))).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    image = image.resize((32, 32), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def _array(path: Path) -> torch.Tensor:
    values = np.asarray(np.load(path))
    tensor = torch.from_numpy(values).float()
    if tensor.ndim != 4:
        raise ValueError(f"expected 4D SSBA array, got {tuple(tensor.shape)}")
    if tuple(tensor.shape[1:]) == (32, 32, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    elif tuple(tensor.shape[1:]) != (3, 32, 32):
        raise ValueError(f"expected [N,3,32,32] or [N,32,32,3] SSBA array, got {tuple(tensor.shape)}")
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).contiguous()


def _load_state(path: Path, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


def _find(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        matches = sorted(root.rglob(name)) if root.exists() else []
        if matches:
            return matches[0]
    return None


def _load_inputaware(path: Path, root: Path, device: torch.device) -> TriggerAdapter:
    root_string = str(root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from attack.inputaware import InputAwareGenerator, Threshold

    artifact: Any = _load_state(path, device)
    args = SimpleNamespace(dataset="cifar10", input_channel=3)
    generator = InputAwareGenerator(args).to(device)
    mask = InputAwareGenerator(args, out_channels=1).to(device)
    generator_state = artifact.get("netG", artifact.get("generator"))
    mask_state = artifact.get("netM", artifact.get("mask"))
    if generator_state is None or mask_state is None:
        raise ValueError("official Input-Aware netCGM.pt must contain netG and netM")
    generator.load_state_dict({key.removeprefix("module."): value for key, value in generator_state.items()})
    mask.load_state_dict({key.removeprefix("module."): value for key, value in mask_state.items()})
    return InputAwareTrigger(TriggerStatus("inputaware", str(path), True), generator, mask, Threshold().to(device))


def _load_ssba_encoder(
    encoder_path: Path,
    provenance_path: Path,
    backdoorbench_root: Path,
    device: torch.device,
) -> TriggerAdapter:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    use_modulated = int(provenance.get("use_modulated", 0))
    resource_root = backdoorbench_root / "resource" / "ssba"
    if str(resource_root) not in sys.path:
        sys.path.insert(0, str(resource_root))
    source = resource_root / ("models_modulated.py" if use_modulated else "models.py")
    if not source.is_file():
        raise FileNotFoundError(f"official SSBA model definition missing: {source}")
    spec = importlib.util.spec_from_file_location("stage1d_ssba_models", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official SSBA encoder from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolution = int(provenance.get("image_resolution", 32))
    channels = int(provenance.get("image_channels", 3))
    length = int(provenance["fingerprint_length"])
    constructor_kwargs = {
        "fingerprint_size": length,
        "return_residual": int(provenance.get("use_residual", 0)),
    }
    if use_modulated:
        constructor_kwargs.update({
            "bias_init": provenance.get("bias_init"),
            "fused_modconv": int(provenance.get("fused_conv", 0)),
            "demodulate": int(provenance.get("demodulate", 1)),
            "fc_layers": int(provenance.get("fc_layers", 0)),
        })
    encoder = module.StegaStampEncoder(resolution, channels, **constructor_kwargs)
    state = _load_state(encoder_path, torch.device("cpu"))
    encoder.load_state_dict({str(k).removeprefix("module."): v for k, v in state.items()}, strict=True)
    encoder = encoder.to(device).eval()
    return SSBAEncoderTrigger(
        TriggerStatus("ssba", f"{encoder_path};{provenance_path}", True),
        encoder,
        provenance,
        bdb_root=backdoorbench_root,
        device=device,
    )


def build_trigger_adapters(
    *, model_root: Path, backdoorbench_root: Path, explicit: dict[str, Path | None],
    device: torch.device, blended_alpha: float, wanet_s: float, wanet_grid_rescale: float,
) -> dict[str, TriggerAdapter]:
    """Build only officially sourced adapters for all requested attack families."""

    adapters: dict[str, TriggerAdapter] = {}
    badnet = explicit.get("badnet") or backdoorbench_root / "resource/badnet/trigger_image.png"
    blended = explicit.get("blended") or backdoorbench_root / "resource/blended/hello_kitty.jpeg"
    if badnet.is_file():
        adapters["badnet"] = FixedImageTrigger(TriggerStatus("badnet", str(badnet), True), _image(badnet), mode="badnet", alpha=0.0)
    else:
        adapters["badnet"] = UnavailableTrigger(TriggerStatus("badnet", str(badnet), False, "official BadNet trigger missing"))
    if blended.is_file():
        adapters["blended"] = FixedImageTrigger(TriggerStatus("blended", str(blended), True), _image(blended), mode="blended", alpha=blended_alpha)
    else:
        adapters["blended"] = UnavailableTrigger(TriggerStatus("blended", str(blended), False, "official Blended trigger missing"))

    wanet_dir = model_root / "wanet" / "seed0"
    identity = explicit.get("wanet_identity") or wanet_dir / "state_identity_grid.pt"
    noise = explicit.get("wanet_noise") or wanet_dir / "state_noise_grid.pt"
    state_path = wanet_dir / "state_dict.pt"
    if identity.is_file() and noise.is_file():
        try:
            adapters["wanet"] = WaNetTrigger(TriggerStatus("wanet", f"{identity};{noise}", True), _load_state(identity, device).float(), _load_state(noise, device).float(), s=wanet_s, grid_rescale=wanet_grid_rescale)
        except Exception as exc:
            adapters["wanet"] = UnavailableTrigger(TriggerStatus("wanet", str(identity), False, str(exc)))
    elif state_path.is_file():
        try:
            state = _load_state(state_path, device)
            adapters["wanet"] = WaNetTrigger(TriggerStatus("wanet", str(state_path), True), state["identity_grid"].float(), state["noise_grid"].float(), s=wanet_s, grid_rescale=wanet_grid_rescale)
        except Exception as exc:
            adapters["wanet"] = UnavailableTrigger(TriggerStatus("wanet", str(state_path), False, str(exc)))
    else:
        adapters["wanet"] = UnavailableTrigger(TriggerStatus("wanet", str(identity), False, "official WaNet grid state missing"))

    ssba_encoder = explicit.get("ssba_encoder")
    ssba_config = explicit.get("ssba_config")
    if ssba_encoder and ssba_config and ssba_encoder.is_file() and ssba_config.is_file():
        try:
            adapters["ssba"] = _load_ssba_encoder(ssba_encoder, ssba_config, backdoorbench_root, device)
        except Exception as exc:
            adapters["ssba"] = UnavailableTrigger(TriggerStatus("ssba", str(ssba_encoder), False, str(exc)))
    else:
        adapters["ssba"] = UnavailableTrigger(TriggerStatus(
            "ssba",
            str(ssba_encoder) if ssba_encoder else None,
            False,
            "official SSBA encoder and provenance config are required for CIFAR-100 generation",
        ))

    inputaware = explicit.get("inputaware") or model_root / "inputaware" / "seed0" / "netCGM.pt"
    if inputaware.is_file():
        try:
            adapters["inputaware"] = _load_inputaware(inputaware, backdoorbench_root, device)
        except Exception as exc:
            adapters["inputaware"] = UnavailableTrigger(TriggerStatus("inputaware", str(inputaware), False, str(exc)))
    else:
        adapters["inputaware"] = UnavailableTrigger(TriggerStatus("inputaware", str(inputaware), False, "official Input-Aware netCGM.pt missing"))

    adaptive = explicit.get("adaptive_blend")
    if adaptive and adaptive.is_file():
        adapters["adaptive_blend"] = FixedImageTrigger(TriggerStatus("adaptive_blend", str(adaptive), True), _image(adaptive), mode="adaptive_blend", alpha=blended_alpha)
    else:
        adapters["adaptive_blend"] = UnavailableTrigger(TriggerStatus("adaptive_blend", str(adaptive) if adaptive else None, False, "official Adaptive-Blend trigger missing"))
    return adapters
