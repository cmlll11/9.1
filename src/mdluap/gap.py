"""Minimal official-loss image-dependent GAP training utilities."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    """Set the generator-training random seed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_official_gap_generator(
    *, gap_root: str, device: torch.device, ngf: int, output_channels: int = 3
) -> nn.Module:
    """Construct and initialize the pinned official GAP ResNet generator."""

    root = str(Path(gap_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from material.models.generators import ResnetGenerator, weights_init

    if device.type != "cuda":
        raise RuntimeError("the official GAP generator is intended for a CUDA run")
    generator = ResnetGenerator(
        3,
        int(output_channels),
        int(ngf),
        norm_type="batch",
        act_type="relu",
        gpu_ids=[device.index or 0],
    )
    generator.apply(weights_init)
    return generator


def train_residual_gap(
    *,
    model: nn.Module,
    mapping: nn.Module,
    train_loader,
    attack_goal: str,
    target_label: int,
    epochs: int,
    lr: float,
    device: torch.device,
    checkpoint_path: str | None = None,
    max_batches: int = 50,
) -> dict:
    """Train official targeted or least-likely non-targeted GAP."""

    if attack_goal not in {"targeted", "non_targeted"}:
        raise ValueError("attack_goal must be targeted or non_targeted")
    optimizer = torch.optim.Adam(mapping.parameters(), lr=float(lr), betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    history: list[float] = []
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mapping.load_state_dict(state["mapping"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        history = list(state.get("history", []))

    model.eval()
    mapping.train()
    for epoch in range(start_epoch, int(epochs)):
        losses = []
        for batch_index, (images, _labels) in enumerate(train_loader):
            if batch_index >= int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                clean_logits = model(images)
                if attack_goal == "targeted":
                    attack_labels = torch.full(
                        (images.shape[0],), int(target_label), dtype=torch.long, device=device
                    )
                else:
                    attack_labels = clean_logits.argmin(dim=1)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(mapping(images)), attack_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        history.append(float(np.mean(losses)) if losses else float("nan"))
        if checkpoint_path:
            tmp_path = f"{checkpoint_path}.tmp"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "mapping": mapping.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
                tmp_path,
            )
            Path(tmp_path).replace(checkpoint_path)
    return {"history": history, "epoch": int(epochs)}
