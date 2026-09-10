"""Evaluate the official Stage 1D models against the fixed quality gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar10_dataset
from mdluap.models import load_model_checkpoint, load_backdoor_toolbox_resnet18


def load_official_datasets(result_path: Path, root: Path):
    """Reconstruct BackdoorBench's stored test-time poisoned dataset."""

    resolved_root = str(root.resolve())
    if resolved_root not in sys.path:
        sys.path.insert(0, resolved_root)
    from utils.save_load_attack import load_attack_result

    previous = os.getcwd()
    os.chdir(resolved_root)
    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    try:
        datasets = load_attack_result(str(result_path))
        for key in ("bd_train", "bd_test"):
            dataset = datasets.get(key)
            if dataset is None:
                continue
            container = dataset.wrapped_dataset.bd_data_container
            for item in container.data_dict.values():
                if isinstance(item, dict) and "path" in item:
                    item["path"] = str(Path(item["path"]).resolve())
        return datasets
    finally:
        torch.load = original_torch_load
        os.chdir(previous)


@torch.inference_mode()
def accuracy(model, dataset, *, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    correct = total = 0
    for images, labels, *rest in loader:
        images, labels = images.to(device), labels.to(device)
        correct += int((model(images).argmax(dim=1) == labels).sum())
        total += int(labels.numel())
    return correct / max(total, 1)


def result_path(model_root: Path, group: str, seed: int) -> Path:
    if group == "clean_select_shared":
        clean_path = model_root / group / f"seed{seed}" / "clean_model.pth"
        if clean_path.is_file():
            return clean_path
    return model_root / group / f"seed{seed}" / "attack_result.pt"


def parse_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--clean-seeds", default="0,1,2,3")
    parser.add_argument("--backdoor-groups", default="badnet,blended,wanet,ssba,inputaware,adaptive_blend")
    parser.add_argument("--backdoor-seed", type=int, default=0)
    parser.add_argument("--adaptive-blend-root", default=None)
    parser.add_argument("--adaptive-blend-model-path", default=None)
    parser.add_argument("--adaptive-blend-trigger-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    bdb_root = Path(args.backdoorbench_root)
    model_root = Path(args.model_root)
    clean_test = cifar10_dataset(data_root, train=False)
    rows: dict[str, dict] = {}

    clean_wrappers = {}
    for seed in parse_csv(args.clean_seeds):
        path = result_path(model_root, args.clean_group, seed)
        wrapper, metadata = load_model_checkpoint(str(path), backdoorbench_root=str(bdb_root), device=device)
        clean_wrappers[seed] = wrapper
        clean_acc = accuracy(wrapper, clean_test, device=device)
        rows[f"{args.clean_group}_seed{seed}"] = {
            "group": args.clean_group, "seed": seed, "result": str(path.resolve()),
            "model_name": metadata["model_name"], "clean_accuracy": clean_acc,
            "status": "qualified" if clean_acc >= 0.90 else "gate_failed",
        }

    clean0 = clean_wrappers[0]
    for group in [item.strip() for item in args.backdoor_groups.split(",") if item.strip()]:
        seed = int(args.backdoor_seed)
        if group == "adaptive_blend":
            model_path = Path(args.adaptive_blend_model_path or model_root / group / f"seed{seed}" / "official_model.pt")
            wrapper, metadata = load_backdoor_toolbox_resnet18(
                str(model_path), backdoor_toolbox_root=str(Path(args.adaptive_blend_root)), device=device
            )
            trigger_path = Path(args.adaptive_blend_trigger_path or Path(args.adaptive_blend_root) / "triggers" / "hellokitty_32.png")
            if not trigger_path.is_file():
                rows[f"{group}_seed{seed}"] = {
                    "group": group, "seed": seed, "result": str(model_path.resolve()),
                    "model_name": metadata["model_name"], "clean_accuracy": accuracy(wrapper, clean_test, device=device),
                    "backdoor_clean_accuracy": accuracy(wrapper, clean_test, device=device),
                    "backdoor_asr": None, "clean_trigger_asr": None,
                    "status": "gate_failed", "reason": "official Adaptive-Blend trigger missing",
                }
                continue
            from PIL import Image
            from torchvision.transforms import Resize, ToTensor
            trigger = Resize((32, 32))(ToTensor()(Image.open(trigger_path).convert("RGB"))).to(device)

            class Triggered(torch.utils.data.Dataset):
                def __init__(self, base): self.base = base
                def __len__(self): return len(self.base)
                def __getitem__(self, index):
                    image, _ = self.base[index]
                    return (0.8 * image + 0.2 * trigger.cpu()).clamp(0, 1), 0

            bd_asr = accuracy(wrapper, Triggered(clean_test), device=device)
            clean_trigger_asr = accuracy(clean0, Triggered(clean_test), device=device)
            clean_acc = accuracy(wrapper, clean_test, device=device)
            passed = clean_acc >= 0.90 and bd_asr >= 0.90 and clean_trigger_asr <= 0.10
            rows[f"{group}_seed{seed}"] = {
                "group": group, "seed": seed, "result": str(model_path.resolve()),
                "model_name": metadata["model_name"], "clean_accuracy": clean_acc,
                "backdoor_clean_accuracy": clean_acc, "backdoor_asr": bd_asr,
                "clean_trigger_asr": clean_trigger_asr, "status": "qualified" if passed else "gate_failed",
            }
            continue

        path = result_path(model_root, group, seed)
        wrapper, metadata = load_attack_result_model(str(path), backdoorbench_root=str(bdb_root), device=device)
        datasets = load_official_datasets(path, bdb_root)
        bd_test = datasets["bd_test"]
        clean_acc = accuracy(wrapper, clean_test, device=device)
        bd_asr = accuracy(wrapper.model, bd_test, device=device)
        clean_trigger_asr = accuracy(clean0.model, bd_test, device=device)
        passed = clean_acc >= 0.90 and bd_asr >= 0.90 and clean_trigger_asr <= 0.10
        rows[f"{group}_seed{seed}"] = {
            "group": group, "seed": seed, "result": str(path.resolve()),
            "model_name": metadata["model_name"], "clean_accuracy": clean_acc,
            "backdoor_clean_accuracy": clean_acc, "backdoor_asr": bd_asr,
            "clean_trigger_asr": clean_trigger_asr, "status": "qualified" if passed else "gate_failed",
        }

    payload = {
        "protocol": "stage1d-official-model-gates-v1",
        "gates": {"clean_accuracy_min": 0.90, "backdoor_clean_accuracy_min": 0.90,
                  "backdoor_asr_min": 0.90, "clean_trigger_asr_max": 0.10},
        "models": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [key for key, row in rows.items() if row["status"] != "qualified"]
    print(json.dumps({"output": str(output.resolve()), "failed": failed}, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
