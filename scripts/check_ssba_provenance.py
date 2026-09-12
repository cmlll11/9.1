"""Verify that the SSBA encoder reproduces the official CIFAR-10 test array."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdluap.official_triggers import (
    _quantize_like_torchvision_save_image,
    build_trigger_adapters,
)


def cifar10_test_images(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    values = np.asarray(payload[b"data"], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
    return torch.from_numpy(values)


def canonical_reference(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path))
    if values.ndim != 4:
        raise ValueError(f"reference SSBA array must be 4D, got {values.shape}")
    if tuple(values.shape[1:]) == (3, 32, 32):
        values = values.transpose(0, 2, 3, 1)
    elif tuple(values.shape[1:]) != (32, 32, 3):
        raise ValueError(f"reference SSBA array has unexpected shape {values.shape}")
    if values.dtype != np.uint8:
        values = np.clip(np.rint(values * (255.0 if values.max() <= 1.0 else 1.0)), 0, 255).astype(np.uint8)
    return values


def _state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "decoder", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError("SSBA decoder checkpoint must contain a state_dict mapping")
    return {str(key).removeprefix("module."): value for key, value in payload.items()}


def validate_decoder(backdoorbench_root: Path, decoder_path: Path, config: dict) -> dict:
    """Load the official decoder with the exact architecture parameters."""

    required = (
        "fingerprint_length", "image_resolution", "image_channels",
        "use_residual", "use_modulated", "fc_layers", "fused_conv",
        "batch_size", "encode_method",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"SSBA provenance config is missing required fields: {missing}")
    method = str(config["encode_method"]).lower()
    if method == "bch" and not config.get("secret") and not config.get("fingerprint_values"):
        raise ValueError("SSBA BCH provenance requires secret or explicit fingerprint_values")
    if method == "seed" and "seed" not in config and not config.get("fingerprint_values"):
        raise ValueError("SSBA seed provenance requires seed or explicit fingerprint_values")

    source = backdoorbench_root / "resource" / "ssba" / (
        "models_modulated.py" if int(config["use_modulated"]) else "models.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"official SSBA model definition missing: {source}")
    spec = importlib.util.spec_from_file_location("stage1d_ssba_decoder_models", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official SSBA model definition from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoder = module.StegaStampDecoder(
        resolution=int(config["image_resolution"]),
        IMAGE_CHANNELS=int(config["image_channels"]),
        fingerprint_size=int(config["fingerprint_length"]),
    )
    payload = torch.load(decoder_path, map_location="cpu", weights_only=False)
    decoder.load_state_dict(_state_dict(payload), strict=True)
    decoder.eval()
    with torch.inference_mode():
        output = decoder(torch.zeros(
            1,
            int(config["image_channels"]),
            int(config["image_resolution"]),
            int(config["image_resolution"]),
        ))
    expected = (1, int(config["fingerprint_length"]))
    if tuple(output.shape) != expected:
        raise ValueError(f"SSBA decoder output shape {tuple(output.shape)} != {expected}")
    return {"source": str(source.resolve()), "output_shape": list(output.shape)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--decoder-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--original-test-batch", required=True)
    parser.add_argument("--reference-test-array", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    encoder = Path(args.encoder_path)
    decoder = Path(args.decoder_path)
    config_path = Path(args.config_path)
    original = Path(args.original_test_batch)
    reference = Path(args.reference_test_array)
    for path in (encoder, decoder, config_path, original, reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder_validation = validate_decoder(Path(args.backdoorbench_root), decoder, config)
    device = torch.device(args.device)
    adapters = build_trigger_adapters(
        model_root=Path(args.backdoorbench_root),
        backdoorbench_root=Path(args.backdoorbench_root),
        explicit={"ssba_encoder": encoder, "ssba_config": config_path},
        device=device,
        blended_alpha=0.2,
        wanet_s=0.5,
        wanet_grid_rescale=1.0,
    )
    adapter = adapters["ssba"]
    if not adapter.status.available:
        raise RuntimeError(adapter.status.reason or "SSBA adapter unavailable")
    images = cifar10_test_images(original)
    generated = []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            end = min(start + args.batch_size, len(images))
            batch = images[start:end].to(device)
            generated.append(adapter.apply(batch, sample_indices=list(range(start, end)), split="cifar100_test").cpu())
    # adapter.apply() already follows the official save_image quantization.
    # Apply the same conversion once more here to obtain the uint8 array that
    # the official PNG -> NPY packing step stores.
    generated_float = torch.cat(generated).permute(0, 2, 3, 1)
    generated_np = (
        _quantize_like_torchvision_save_image(generated_float)
        * 255.0
    ).to(torch.uint8).numpy()
    reference_np = canonical_reference(reference)
    if generated_np.shape != reference_np.shape:
        raise ValueError(f"SSBA shape mismatch: generated={generated_np.shape}, reference={reference_np.shape}")
    diff = np.abs(generated_np.astype(np.int16) - reference_np.astype(np.int16))
    result = {
        "encoder_path": str(encoder.resolve()),
        "decoder_path": str(decoder.resolve()),
        "config_path": str(config_path.resolve()),
        "reference_path": str(reference.resolve()),
        "sample_count": int(len(reference_np)),
        "exact_match": bool(np.array_equal(generated_np, reference_np)),
        "max_abs_pixel_diff": int(diff.max()),
        "different_pixel_count": int(np.count_nonzero(diff)),
        "mean_abs_pixel_diff": float(diff.mean()),
        "positive_diff_count": int(np.count_nonzero(generated_np.astype(np.int16) > reference_np.astype(np.int16))),
        "negative_diff_count": int(np.count_nonzero(generated_np.astype(np.int16) < reference_np.astype(np.int16))),
        "quantization": "torchvision_save_image_floor_255x_plus_half_to_uint8",
        "config": config,
        "decoder_validation": decoder_validation,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["exact_match"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
