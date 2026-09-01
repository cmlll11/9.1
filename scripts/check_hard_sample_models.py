"""Check classifier quality for the multi-seed hard-sample experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mdluap.data import cifar10_dataset
from mdluap.models import load_attack_result_model


def load_official_datasets(result_path: Path, root: str):
    resolved_root = str(Path(root).resolve())
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
        # BackdoorBench stores generated PNG paths relative to its root.
        # Make them absolute before restoring the caller's working directory.
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
def accuracy(model, dataset, *, device: torch.device, workers: int) -> float:
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=workers)
    correct = total = 0
    for batch in data_loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        correct += int((model(images).argmax(dim=1) == labels).sum())
        total += int(labels.numel())
    return correct / max(total, 1)


def result_path(root: Path, group: str, seed: int) -> Path:
    return root / group / f"seed{seed}" / "attack_result.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--selection-group", default="clean_select")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    root = Path(args.model_root)
    clean_eval_path = result_path(root, "clean_eval", 5)
    clean_wrapper, _ = load_attack_result_model(
        str(clean_eval_path), backdoorbench_root=args.backdoorbench_root, device=device
    )
    clean_model = clean_wrapper.model
    clean_test = cifar10_dataset(args.data_root, train=False)
    rows = {}

    for group, seeds in (("clean_select", range(5)), ("clean_eval", range(5, 10))):
        for seed in seeds:
            artifact_group = args.selection_group if group == "clean_select" else group
            path = result_path(root, artifact_group, seed)
            wrapper, metadata = load_attack_result_model(
                str(path), backdoorbench_root=args.backdoorbench_root, device=device
            )
            clean_accuracy = accuracy(wrapper, clean_test, device=device, workers=args.workers)
            rows[f"{group}_seed{seed}"] = {
                "group": group,
                "seed": seed,
                "result": str(path.resolve()),
                "model_name": metadata["model_name"],
                "clean_accuracy": clean_accuracy,
                "status": "qualified" if clean_accuracy >= 0.90 else "gate_failed",
            }

    for trigger in ("badnet", "lf", "blended", "wanet"):
        for seed in range(5):
            path = result_path(root, trigger, seed)
            wrapper, metadata = load_attack_result_model(
                str(path), backdoorbench_root=args.backdoorbench_root, device=device
            )
            official_path = (
                Path(args.backdoorbench_root)
                / "record"
                / f"mdl_uap_hard_{trigger}_seed{seed}"
                / "attack_result.pt"
            )
            datasets = load_official_datasets(official_path, args.backdoorbench_root)
            backdoor_clean_accuracy = accuracy(wrapper, clean_test, device=device, workers=args.workers)
            backdoor_asr = accuracy(wrapper.model, datasets["bd_test"], device=device, workers=args.workers)
            clean_trigger_asr = accuracy(clean_model, datasets["bd_test"], device=device, workers=args.workers)
            passed = bool(
                backdoor_clean_accuracy >= 0.90
                and backdoor_asr >= 0.90
                and clean_trigger_asr <= 0.10
            )
            rows[f"{trigger}_seed{seed}"] = {
                "group": trigger,
                "seed": seed,
                "result": str(path.resolve()),
                "model_name": metadata["model_name"],
                "clean_accuracy": backdoor_clean_accuracy,
                "backdoor_clean_accuracy": backdoor_clean_accuracy,
                "backdoor_asr": backdoor_asr,
                "clean_trigger_asr": clean_trigger_asr,
                "status": "qualified" if passed else "gate_failed",
            }
            print(
                f"{trigger} seed={seed}: clean_acc={backdoor_clean_accuracy:.4f} "
                f"asr={backdoor_asr:.4f} clean_trigger={clean_trigger_asr:.4f} "
                f"status={rows[f'{trigger}_seed{seed}']['status']}",
                flush=True,
            )

    report = {
        "protocol": "hard-sample-model-gates-v1",
        "gates": {
            "clean_accuracy_min": 0.90,
            "backdoor_clean_accuracy_min": 0.90,
            "backdoor_asr_min": 0.90,
            "clean_trigger_asr_max": 0.10,
        },
        "models": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model gates complete: {output.resolve()}")


if __name__ == "__main__":
    main()
