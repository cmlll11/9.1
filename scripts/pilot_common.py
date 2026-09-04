"""Shared utilities for the Stage 1A/1B pilot scripts."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mdluap.models import load_attack_result_model


def seed_everything(seed: int) -> None:
    """Set Python, NumPy and Torch seeds used by deterministic selection/attacks."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def timestamp_run_dir(root: str | Path, stage: str) -> Path:
    """Create a unique timestamped result directory without overwriting runs."""

    base = Path(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = base / f"{stage}_{run_id}"
    suffix = 0
    while output.exists():
        suffix += 1
        output = base / f"{stage}_{run_id}_{suffix:02d}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    return output


def write_json(path: str | Path, payload: object) -> None:
    """Write human-readable UTF-8 JSON and create missing parent directories."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Write a stable CSV using the union of row keys as the column schema."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_model(path: str | Path, backdoorbench_root: str, device: torch.device):
    """Load a frozen normalized BackdoorBench classifier with an existence check."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return load_attack_result_model(str(path), backdoorbench_root=backdoorbench_root, device=device)


def batch_images(dataset: Dataset, indices: list[int], *, batch_size: int, device: torch.device):
    """Yield raw ``[N, C, H, W]`` tensors and dataset indices in fixed order."""

    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        images = torch.stack([dataset[index][0] for index in batch_indices]).to(device, non_blocking=True)
        labels = torch.tensor([int(dataset[index][1]) for index in batch_indices], device=device)
        yield batch_indices, images, labels


@torch.inference_mode()
def logits_for_indices(model, dataset: Dataset, indices: list[int], *, batch_size: int, device: torch.device):
    """Compute logits and predictions for indexed raw images in stable order."""

    logits_rows: list[Tensor] = []
    labels_rows: list[Tensor] = []
    for _, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits_rows.append(model(images).detach().cpu())
        labels_rows.append(labels.detach().cpu())
    return torch.cat(logits_rows), torch.cat(labels_rows)


def load_partition_indices(partition_root: str | Path, partition: str = "shared") -> list[int]:
    """Load original CIFAR-10 indices from the existing partition metadata."""

    # prepare_hard_sample_partitions.py stores the original indices in
    # partition.json next to each synthetic CIFAR-10 directory.
    metadata = Path(partition_root) / partition / "partition.json"
    if not metadata.is_file():
        raise FileNotFoundError(
            f"cannot identify actual training samples; expected partition metadata at {metadata}"
        )
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"partition metadata has no indices: {metadata}")
    return [int(index) for index in indices]

