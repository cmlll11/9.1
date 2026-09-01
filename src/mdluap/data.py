"""Small CIFAR-10 data helpers for raw-pixel UAP training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def cifar10_dataset(root: str, *, train: bool) -> Dataset:
    """Return CIFAR-10 tensors in [0, 1]; model normalization is done later."""

    dataset_root = Path(root)
    if dataset_root.name != "cifar10":
        dataset_root = dataset_root / "cifar10"
    return datasets.CIFAR10(
        root=str(dataset_root),
        train=train,
        transform=transforms.ToTensor(),
        download=False,
    )


def cifar10_split(root: str, *, split: str, split_seed: int = 2026, val_size: int = 5000) -> Dataset:
    """Return the fixed mapping-train, validation, or untouched test split."""

    if split == "test":
        return cifar10_dataset(root, train=False)
    if split not in {"train", "val"}:
        raise ValueError("split must be train, val, or test")
    full_train = cifar10_dataset(root, train=True)
    generator = torch.Generator().manual_seed(int(split_seed))
    order = torch.randperm(len(full_train), generator=generator).tolist()
    # Keep the historical 5,000-image validation split for full CIFAR-10,
    # but avoid an empty training split for small hard-sample partitions.
    effective_val_size = min(int(val_size), max(1, len(full_train) // 5))
    indices = order[:-effective_val_size] if split == "train" else order[-effective_val_size:]
    return Subset(full_train, indices)


def loader(dataset: Dataset, *, batch_size: int, train: bool, workers: int) -> DataLoader:
    """Build a deterministic evaluation loader or a shuffled training loader."""

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=train,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
