"""Stage 1D: targeted-robust samples and trigger-direction alignment.

This script is a mechanism experiment, not a deployment detector.  For each
Clean seed it first selects CIFAR-100 samples that are difficult to move to
target class 0 with targeted PGD.  The same seed-matched images are then
evaluated on the Clean and BadNet models.  It compares the feature change
caused by a real BadNet trigger with the feature change caused by targeted
PGD.

All images passed to the models are raw tensors in [0, 1].  The model wrapper
performs the CIFAR-10 normalization used during BackdoorBench training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar100_dataset
from mdluap.targeted_pgd import FINE_EPSILON_GRID, targeted_pgd, targeted_pgd_endpoint
from pilot_common import batch_images, load_model, seed_everything, timestamp_run_dir, write_csv, write_json


SELECTION_EPS_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
ANALYSIS_EPS_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
FIXED_REPORT_EPS_PIXELS = (1.0, 1.5, 2.0)


def parse_ints(value: str) -> list[int]:
    """Parse a comma-separated integer argument."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> tuple[float, ...]:
    """Parse and validate an ascending comma-separated pixel grid."""

    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(value <= 0 for value in values) or tuple(sorted(values)) != values:
        raise ValueError("epsilon grid must be a non-empty ascending list of positive numbers")
    return values


def build_trigger_transforms(
    *,
    backdoorbench_root: Path,
    model_root: Path,
    groups: list[str],
    seed: int,
    badnet_path: Path | None,
    lf_path: Path | None,
    blended_path: Path | None,
    wanet_state_path: Path | None,
    blended_alpha: float,
    wanet_s: float,
    wanet_grid_rescale: float,
    device: torch.device,
) -> tuple[dict[str, callable], dict[str, str]]:
    """Construct the trigger transform belonging to each requested model.

    Clean uses the BadNet patch as a fixed trigger control.  Each backdoor
    model uses its own training-time trigger: LF additive noise, Blended
    alpha-compositing, or the exact WaNet grids saved with the model.
    """

    transforms_by_group: dict[str, callable] = {}
    source_by_group: dict[str, str] = {}

    badnet_path = badnet_path or backdoorbench_root / "resource" / "badnet" / "trigger_image.png"
    if "clean" in groups or "badnet" in groups:
        badnet_trigger = load_badnet_trigger(badnet_path)
        for group in ("clean", "badnet"):
            if group in groups:
                transforms_by_group[group] = lambda images, trigger=badnet_trigger: apply_badnet_trigger(images, trigger)
                source_by_group[group] = str(badnet_path)

    if "lf" in groups:
        lf_path = lf_path or backdoorbench_root / "resource" / "lowFrequency" / "cifar10_preactresnet18_0_255.npy"
        lf_trigger = load_lf_trigger(lf_path)
        transforms_by_group["lf"] = lambda images, trigger=lf_trigger: apply_lf_trigger(images, trigger)
        source_by_group["lf"] = str(lf_path)

    if "blended" in groups:
        blended_path = blended_path or backdoorbench_root / "resource" / "blended" / "hello_kitty.jpeg"
        blended_trigger = load_blended_trigger(blended_path)
        transforms_by_group["blended"] = lambda images, trigger=blended_trigger: apply_blended_trigger(
            images, trigger, alpha=blended_alpha
        )
        source_by_group["blended"] = str(blended_path)

    if "wanet" in groups:
        wanet_state_path = wanet_state_path or model_root / "wanet" / f"seed{seed}" / "state_dict.pt"
        identity_grid, noise_grid = load_wanet_grids(wanet_state_path, device=device)
        transforms_by_group["wanet"] = lambda images, identity=identity_grid, noise=noise_grid: apply_wanet_trigger(
            images, identity, noise, s=wanet_s, grid_rescale=wanet_grid_rescale
        )
        source_by_group["wanet"] = str(wanet_state_path)

    missing = [group for group in groups if group not in transforms_by_group]
    if missing:
        raise ValueError(f"unsupported backdoor groups for Stage 1D: {missing}")
    return transforms_by_group, source_by_group


def checkpoint_path(model_root: Path, group: str, seed: int) -> Path:
    """Resolve the packaged BackdoorBench checkpoint path."""

    return model_root / group / f"seed{seed}" / "attack_result.pt"


class AvgPoolFeatureExtractor:
    """Capture the 512-dimensional PreActResNet18 avgpool representation."""

    def __init__(self, classifier: torch.nn.Module):
        backbone = getattr(classifier, "model", None)
        layer = getattr(backbone, "avgpool", None)
        if layer is None:
            raise AttributeError("the checkpoint model has no model.avgpool feature layer")
        self.output: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._hook)
        self.classifier = classifier

    def _hook(self, _module, _inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.output = output

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and flattened features for raw images."""

        logits = self.classifier(images)
        if self.output is None:
            raise RuntimeError("avgpool hook did not capture a feature tensor")
        features = self.output.flatten(1).detach()
        return logits.detach(), features

    def close(self) -> None:
        """Remove the forward hook."""

        self.handle.remove()


def load_badnet_trigger(path: str | Path) -> torch.Tensor:
    """Load the official BadNet patch as a raw [3, 32, 32] tensor.

    BackdoorBench treats positive trigger pixels as replacement values and
    zero pixels as mask locations.  The same rule is applied below to raw
    CIFAR images in [0, 1].
    """

    image = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def apply_badnet_trigger(images: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    """Apply the official mask-patch rule to a batch of raw images."""

    trigger = trigger.to(device=images.device, dtype=images.dtype).unsqueeze(0)
    mask = trigger > 0
    return torch.where(mask, trigger, images)


def load_lf_trigger(path: str | Path) -> torch.Tensor:
    """Load BackdoorBench's CIFAR-10 low-frequency additive pattern.

    BackdoorBench applies this pattern in uint8 image space with values in
    ``[0, 255]``.  The experiment keeps images in ``[0, 1]``, so the pattern
    is converted to that same scale here.
    """

    array = np.load(path)
    if array.shape != (32, 32, 3):
        raise ValueError(f"LF trigger must have shape (32, 32, 3), got {array.shape}")
    return torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()


def apply_lf_trigger(images: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    """Apply BackdoorBench's clipped uint8-equivalent LF additive trigger."""

    trigger = trigger.to(device=images.device, dtype=images.dtype).unsqueeze(0)
    return torch.clamp(images + trigger, 0.0, 1.0)


def load_blended_trigger(path: str | Path) -> torch.Tensor:
    """Load and resize the official Blended trigger image to ``[3, 32, 32]``."""

    image = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def apply_blended_trigger(
    images: torch.Tensor,
    trigger: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Apply BackdoorBench's test-time alpha blending in raw image space."""

    trigger = trigger.to(device=images.device, dtype=images.dtype).unsqueeze(0)
    return torch.clamp((1.0 - float(alpha)) * images + float(alpha) * trigger, 0.0, 1.0)


def load_wanet_grids(path: str | Path, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the exact WaNet grids saved by BackdoorBench during training.

    The WaNet training command used by this repository saves ``state_dict.pt``
    every ten epochs, including the final checkpoint.  That file contains the
    ``identity_grid`` and ``noise_grid`` needed to reproduce the learned
    geometric trigger on arbitrary images.
    """

    state = torch.load(path, map_location=device, weights_only=False)
    if "identity_grid" not in state or "noise_grid" not in state:
        raise KeyError(f"WaNet state must contain identity_grid and noise_grid: {path}")
    identity_grid = state["identity_grid"].to(device=device, dtype=torch.float32)
    noise_grid = state["noise_grid"].to(device=device, dtype=torch.float32)
    if tuple(identity_grid.shape) != (1, 32, 32, 2) or tuple(noise_grid.shape) != (1, 32, 32, 2):
        raise ValueError(
            "WaNet grids must both have shape (1, 32, 32, 2), "
            f"got identity={tuple(identity_grid.shape)}, noise={tuple(noise_grid.shape)}"
        )
    return identity_grid, noise_grid


def apply_wanet_trigger(
    images: torch.Tensor,
    identity_grid: torch.Tensor,
    noise_grid: torch.Tensor,
    *,
    s: float,
    grid_rescale: float,
) -> torch.Tensor:
    """Apply the deterministic WaNet grid used for ordinary backdoor samples."""

    grid = (identity_grid + float(s) * noise_grid / images.shape[-2]) * float(grid_rescale)
    grid = torch.clamp(grid, -1.0, 1.0).expand(images.shape[0], -1, -1, -1)
    return F.grid_sample(images, grid, align_corners=True)


@torch.inference_mode()
def logits_for_indices(model, dataset, indices: list[int], *, batch_size: int, device: torch.device):
    """Return logits, predictions and labels in the requested index order."""

    logits_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for _, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits_rows.append(model(images).detach().cpu())
        label_rows.append(labels.detach().cpu())
    logits = torch.cat(logits_rows)
    labels = torch.cat(label_rows)
    return logits, logits.argmax(dim=1), labels


def run_radius_grid(
    model,
    dataset,
    indices: list[int],
    *,
    target: int,
    eps_pixels: Iterable[float],
    steps: int,
    random_start: bool,
    restarts: int,
    batch_size: int,
    device: torch.device,
    model_group: str,
    seed: int,
    phase: str,
) -> tuple[list[dict], dict[int, float]]:
    """Run a targeted PGD grid and return long records plus first radii."""

    before_logits, before_predictions, _ = logits_for_indices(model, dataset, indices, batch_size=batch_size, device=device)
    before = before_predictions.tolist()
    found = {index: prediction == int(target) for index, prediction in zip(indices, before)}
    radius = {index: 0.0 if prediction == int(target) else float("inf") for index, prediction in zip(indices, before)}
    rows: list[dict] = []

    for epsilon_pixels in eps_pixels:
        epsilon = float(epsilon_pixels) / 255.0
        success_parts: list[torch.Tensor] = []
        norm_parts: list[torch.Tensor] = []
        prediction_parts: list[torch.Tensor] = []
        for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
            result = targeted_pgd(
                model,
                images,
                target=target,
                epsilon=epsilon,
                steps=steps,
                alpha=epsilon / 10.0,
                random_start=random_start,
                restarts=restarts,
            )
            success_parts.append(result.success.detach().cpu())
            norm_parts.append(result.best_linf.detach().cpu())
            prediction_parts.append(result.best_prediction.detach().cpu())
        successes = torch.cat(success_parts).tolist()
        norms = torch.cat(norm_parts).tolist()
        predictions = torch.cat(prediction_parts).tolist()
        for offset, sample_index in enumerate(indices):
            success = bool(successes[offset])
            if success and not found[sample_index]:
                radius[sample_index] = float(norms[offset])
                found[sample_index] = True
            rows.append(
                {
                    "model_group": model_group,
                    "seed": int(seed),
                    "phase": phase,
                    "sample_index": int(sample_index),
                    "epsilon_pixels": float(epsilon_pixels),
                    "steps": int(steps),
                    "restarts": int(restarts),
                    "random_start": bool(random_start),
                    "before_prediction": int(before[offset]),
                    "after_prediction": int(predictions[offset]),
                    "success": success,
                    "actual_linf_pixels": float(norms[offset]) * 255.0 if math.isfinite(float(norms[offset])) else None,
                    "target": int(target),
                }
            )
    return rows, radius


def robust_sort(indices: list[int], radius: dict[int, float], before: list[int], target: int, count: int) -> list[int]:
    """Select the largest first-success radius, putting censored rows first."""

    eligible = [index for index, prediction in zip(indices, before) if int(prediction) != int(target)]
    eligible.sort(key=lambda index: (-float(radius.get(index, float("inf"))), int(index)))
    return eligible[: int(count)]


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity, returning NaN for zero vectors."""

    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def rank_average(values: np.ndarray) -> np.ndarray:
    """Return average ranks with deterministic handling of ties."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return Spearman correlation over finite paired observations."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return None
    x = rank_average(left[mask])
    y = rank_average(right[mask])
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_concentration(vectors: np.ndarray) -> float | None:
    """Mean off-diagonal cosine among nonzero finite direction vectors."""

    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[np.isfinite(norms) & (norms > 0)]
    if len(vectors) < 2:
        return None
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    mask = ~np.eye(len(normalized), dtype=bool)
    return float(similarity[mask].mean())


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a permutation with no fixed point for shuffled controls."""

    if size < 2:
        return np.arange(size)
    while True:
        permutation = rng.permutation(size)
        if not np.any(permutation == np.arange(size)):
            return permutation


def shuffled_alignment(adv: np.ndarray, trigger: np.ndarray, *, seed: int, repeats: int = 20) -> np.ndarray:
    """Average adversarial-to-other-sample-trigger alignment over derangements."""

    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(repeats)):
        permutation = derangement(len(trigger), rng)
        values.append(cosine_rows(adv, trigger[permutation]))
    return np.nanmean(np.stack(values, axis=0), axis=0)


def finite_stats(values: np.ndarray) -> dict:
    """Return compact finite mean, median and quartiles for a CSV row."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def optional_float_array(values: Iterable[object]) -> np.ndarray:
    """Convert nullable numeric CSV values to a float array with NaNs."""

    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)


def read_quality_report(path: str | None) -> list[dict]:
    """Read the existing model-gate JSON without reimplementing triggers."""

    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("models", payload.get("records", payload.get("results", [])))
        if isinstance(records, dict):
            records = list(records.values())
    else:
        records = []
    return [dict(record) for record in records if isinstance(record, dict)]


def write_log(output: Path, message: str) -> None:
    """Print and append one progress message to the run-local log."""

    print(message, flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def make_figures(output: Path, rows: list[dict]) -> None:
    """Create the requested compact figures when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    first_rows = [row for row in rows if row["protocol"] == "first_success" and not row["pre_target"]]
    groups: dict[str, list[float]] = {}
    for row in first_rows:
        if row["same_alignment"] is not None and math.isfinite(float(row["same_alignment"])):
            groups.setdefault(f"{row['model_group']}-s{row['seed']}-{row['sample_group']}", []).append(float(row["same_alignment"]))
    if groups:
        plt.figure(figsize=(10, 5))
        plt.boxplot(list(groups.values()), labels=list(groups.keys()), showfliers=False)
        plt.ylabel("same-sample trigger/PGD cosine")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output / "figures" / "alignment_boxplot.png", dpi=160)
        plt.close()

    trigger_values = {}
    adv_values = {}
    for row in first_rows:
        key = f"{row['model_group']}-s{row['seed']}"
        if row["trigger_concentration"] is not None and math.isfinite(float(row["trigger_concentration"])):
            trigger_values.setdefault(key, float(row["trigger_concentration"]))
        if row["sample_group"] == "success" and row["adv_concentration"] is not None and math.isfinite(float(row["adv_concentration"])):
            adv_values.setdefault(key, float(row["adv_concentration"]))
    if trigger_values or adv_values:
        keys = sorted(set(trigger_values) | set(adv_values))
        positions = np.arange(len(keys))
        width = 0.38
        plt.figure(figsize=(10, 5))
        plt.bar(positions - width / 2, [trigger_values.get(key, np.nan) for key in keys], width, label="trigger")
        plt.bar(positions + width / 2, [adv_values.get(key, np.nan) for key in keys], width, label="adv success")
        plt.xticks(positions, keys, rotation=45, ha="right")
        plt.ylabel("pairwise direction cosine")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "figures" / "direction_concentration.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    """Parse server paths and the fixed Stage 1D protocol."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output-root", default="results/stage1d_trigger_alignment")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--backdoor-groups", default="badnet,lf,blended,wanet")
    parser.add_argument("--clean-seeds", default="0")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--candidate-seed", type=int, default=2031)
    parser.add_argument("--top-coarse", type=int, default=300)
    parser.add_argument("--top-final", type=int, default=100)
    parser.add_argument("--selection-eps-pixels", default=",".join(map(str, SELECTION_EPS_PIXELS)))
    parser.add_argument("--analysis-eps-pixels", default=",".join(map(str, ANALYSIS_EPS_PIXELS)))
    parser.add_argument("--fixed-report-eps-pixels", default=",".join(map(str, FIXED_REPORT_EPS_PIXELS)))
    parser.add_argument("--selection-coarse-steps", type=int, default=30)
    parser.add_argument("--selection-refined-steps", type=int, default=100)
    parser.add_argument("--analysis-steps", type=int, default=100)
    parser.add_argument("--selection-refined-restarts", type=int, default=3)
    parser.add_argument("--analysis-restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--trigger-path", default=None)
    parser.add_argument("--lf-trigger-path", default=None)
    parser.add_argument("--blended-trigger-path", default=None)
    parser.add_argument("--wanet-state-path", default=None)
    parser.add_argument("--blended-alpha", type=float, default=0.2)
    parser.add_argument("--wanet-s", type=float, default=0.5)
    parser.add_argument("--wanet-grid-rescale", type=float, default=1.0)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    """Run selection, paired attacks, feature analysis and result export."""

    args = parse_args()
    clean_seeds = parse_ints(args.clean_seeds)
    backdoor_groups = [item.strip() for item in args.backdoor_groups.split(",") if item.strip()]
    if not backdoor_groups:
        raise ValueError("at least one backdoor group is required")
    selection_eps = parse_floats(args.selection_eps_pixels)
    analysis_eps = parse_floats(args.analysis_eps_pixels)
    fixed_eps = parse_floats(args.fixed_report_eps_pixels)
    if not set(fixed_eps).issubset(set(analysis_eps)):
        raise ValueError("fixed report epsilons must be contained in the analysis grid")

    output = timestamp_run_dir(args.output_root, "stage1d")
    device = torch.device(args.device)
    data = cifar100_dataset(args.data_root, train=False)
    model_root = Path(args.model_root)
    backdoorbench_root = Path(args.backdoorbench_root)
    trigger_path = Path(args.trigger_path) if args.trigger_path else backdoorbench_root / "resource" / "badnet" / "trigger_image.png"
    trigger_groups = ["clean"] + backdoor_groups
    trigger_transforms, trigger_sources = build_trigger_transforms(
        backdoorbench_root=backdoorbench_root,
        model_root=model_root,
        groups=trigger_groups,
        seed=clean_seeds[0],
        badnet_path=trigger_path,
        lf_path=Path(args.lf_trigger_path) if args.lf_trigger_path else None,
        blended_path=Path(args.blended_trigger_path) if args.blended_trigger_path else None,
        wanet_state_path=Path(args.wanet_state_path) if args.wanet_state_path else None,
        blended_alpha=args.blended_alpha,
        wanet_s=args.wanet_s,
        wanet_grid_rescale=args.wanet_grid_rescale,
        device=device,
    )
    try:
        import yaml
        resolved_config = {
            "protocol": "stage1d-targeted-robust-trigger-alignment-v1",
            "arguments": vars(args),
            "clean_seeds": clean_seeds,
            "selection_epsilon_pixels": list(selection_eps),
            "analysis_epsilon_pixels": list(analysis_eps),
            "fixed_report_epsilon_pixels": list(fixed_eps),
            "trigger_sources": trigger_sources,
            "blended_alpha": float(args.blended_alpha),
            "wanet_s": float(args.wanet_s),
            "wanet_grid_rescale": float(args.wanet_grid_rescale),
            "feature_layer": "model.avgpool",
            "feature_dimension": 512,
        }
        (output / "config.resolved.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except ImportError:
        (output / "config.resolved.yaml").write_text(
            json.dumps({"arguments": vars(args)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    seed_everything(args.candidate_seed)
    order = np.random.default_rng(args.candidate_seed).permutation(len(data)).tolist()
    candidate_indices = [int(index) for index in order[: int(args.candidate_count)]]
    write_csv(output / "candidate_pool.csv", [
        {"pool_position": position, "sample_index": index, "dataset": "CIFAR100_test"}
        for position, index in enumerate(candidate_indices)
    ])
    write_log(output, f"Stage 1D started: {output.resolve()}")
    write_log(output, f"candidate_count={len(candidate_indices)} clean_seeds={clean_seeds} target={args.target}")

    selection_rows: list[dict] = []
    selected_rows: list[dict] = []
    selected_by_seed: dict[int, list[int]] = {}
    clean_models: dict[int, torch.nn.Module] = {}

    for seed in clean_seeds:
        model, _ = load_model(checkpoint_path(model_root, args.clean_group, seed), str(backdoorbench_root), device)
        clean_models[seed] = model
        seed_everything(args.candidate_seed + seed)
        before_logits, before_predictions, _ = logits_for_indices(model, data, candidate_indices, batch_size=args.batch_size, device=device)
        coarse_rows, coarse_radius = run_radius_grid(
            model, data, candidate_indices, target=args.target, eps_pixels=selection_eps,
            steps=args.selection_coarse_steps, random_start=False, restarts=1,
            batch_size=args.batch_size, device=device, model_group=args.clean_group,
            seed=seed, phase="selection_coarse",
        )
        selection_rows.extend(coarse_rows)
        coarse_top = robust_sort(candidate_indices, coarse_radius, before_predictions.tolist(), args.target, args.top_coarse)
        refined_rows, refined_radius = run_radius_grid(
            model, data, coarse_top, target=args.target, eps_pixels=selection_eps,
            steps=args.selection_refined_steps, random_start=True, restarts=args.selection_refined_restarts,
            batch_size=args.batch_size, device=device, model_group=args.clean_group,
            seed=seed, phase="selection_refined",
        )
        selection_rows.extend(refined_rows)
        coarse_prediction_map = {index: int(prediction) for index, prediction in zip(candidate_indices, before_predictions.tolist())}
        selected = robust_sort(coarse_top, refined_radius, [coarse_prediction_map[index] for index in coarse_top], args.target, args.top_final)
        selected_by_seed[seed] = selected
        for rank, index in enumerate(selected, start=1):
            radius = float(refined_radius.get(index, float("inf")))
            selected_rows.append({
                "selection_clean_seed": seed,
                "rank": rank,
                "sample_index": int(index),
                "refined_radius_pixels": radius * 255.0 if math.isfinite(radius) else None,
                "censored_gt_max_epsilon": not math.isfinite(radius),
            })
        write_log(output, f"seed{seed}: selected {len(selected)} targeted-robust samples")

    write_csv(output / "targeted_selection_records.csv", selection_rows)
    write_csv(output / "selected_targeted_robust_samples.csv", selected_rows)

    attack_rows: list[dict] = []
    trigger_rows: list[dict] = []
    alignment_rows: list[dict] = []
    feature_dimension: int | None = None
    model_specs = [("clean", args.clean_group)] + [(group, group) for group in backdoor_groups]

    for seed in clean_seeds:
        indices = selected_by_seed[seed]
        for model_group, checkpoint_group in model_specs:
            model = clean_models[seed] if model_group == "clean" else load_model(
                checkpoint_path(model_root, checkpoint_group, seed), str(backdoorbench_root), device
            )[0]
            seed_everything(args.candidate_seed + 1000 + seed * 10 + (0 if model_group == "clean" else 1))
            extractor = AvgPoolFeatureExtractor(model)
            try:
                base_feature_parts: list[np.ndarray] = []
                trigger_feature_parts: list[np.ndarray] = []
                base_logit_parts: list[np.ndarray] = []
                trigger_logit_parts: list[np.ndarray] = []
                original_prediction_parts: list[np.ndarray] = []
                target_margin_parts: list[np.ndarray] = []
                endpoint_by_eps: dict[float, list[dict]] = {float(eps): [] for eps in analysis_eps}
                for batch_indices, images, _ in batch_images(data, indices, batch_size=args.batch_size, device=device):
                    base_logits, base_features = extractor(images)
                    triggered_images = trigger_transforms[model_group](images)
                    trigger_logits, trigger_features = extractor(triggered_images)
                    base_np = base_features.cpu().numpy()
                    trigger_np = trigger_features.cpu().numpy()
                    base_logit_np = base_logits.cpu().numpy()
                    trigger_logit_np = trigger_logits.cpu().numpy()
                    base_feature_parts.append(base_np)
                    trigger_feature_parts.append(trigger_np)
                    base_logit_parts.append(base_logit_np)
                    trigger_logit_parts.append(trigger_logit_np)
                    original_prediction_parts.append(base_logits.argmax(dim=1).cpu().numpy())
                    other_logits = base_logits.clone()
                    other_logits[:, args.target] = float("-inf")
                    target_margin_parts.append((other_logits.max(dim=1).values - base_logits[:, args.target]).cpu().numpy())

                    for eps_pixels in analysis_eps:
                        epsilon = float(eps_pixels) / 255.0
                        endpoint_result = targeted_pgd_endpoint(
                            model, images, target=args.target, epsilon=epsilon,
                            steps=args.analysis_steps, alpha=epsilon / 10.0,
                            random_start=True, restarts=args.analysis_restarts,
                        )
                        endpoint_logits, endpoint_features = extractor(endpoint_result.endpoint)
                        endpoint_by_eps[float(eps_pixels)].append({
                            "sample_indices": list(batch_indices),
                            "success": endpoint_result.success.cpu().numpy(),
                            "endpoint_prediction": endpoint_result.endpoint_prediction.cpu().numpy(),
                            "endpoint_linf_pixels": endpoint_result.endpoint_linf.cpu().numpy() * 255.0,
                            "target_loss": endpoint_result.target_loss.cpu().numpy(),
                            "endpoint_features": endpoint_features.cpu().numpy(),
                        })

                base_features = np.concatenate(base_feature_parts, axis=0)
                trigger_features = np.concatenate(trigger_feature_parts, axis=0)
                feature_dimension = int(base_features.shape[1])
                base_logits = np.concatenate(base_logit_parts, axis=0)
                trigger_logits = np.concatenate(trigger_logit_parts, axis=0)
                original_predictions = np.concatenate(original_prediction_parts, axis=0)
                target_margins = np.concatenate(target_margin_parts, axis=0)
                trigger_predictions = trigger_logits.argmax(axis=1)
                pre_target = original_predictions == int(args.target)
                trigger_success = trigger_predictions == int(args.target)
                trigger_delta = trigger_features - base_features
                trigger_concentration_all = pairwise_concentration(trigger_delta[~pre_target])
                trigger_concentration_success = pairwise_concentration(trigger_delta[(~pre_target) & trigger_success])

                trigger_record_by_index = {}
                for position, sample_index in enumerate(indices):
                    row = {
                        "model_group": model_group,
                        "seed": int(seed),
                        "trigger_source": trigger_sources[model_group],
                        "sample_index": int(sample_index),
                        "original_prediction": int(original_predictions[position]),
                        "trigger_prediction": int(trigger_predictions[position]),
                        "trigger_success": bool(trigger_success[position]),
                        "pre_target": bool(pre_target[position]),
                        "target_resistance_margin": float(target_margins[position]),
                        "trigger_feature_norm": float(np.linalg.norm(trigger_delta[position])),
                    }
                    trigger_rows.append(row)
                    trigger_record_by_index[int(sample_index)] = row

                endpoint_arrays: dict[float, dict] = {}
                for eps_pixels in analysis_eps:
                    parts = endpoint_by_eps[float(eps_pixels)]
                    endpoint_arrays[float(eps_pixels)] = {
                        "success": np.concatenate([part["success"] for part in parts]),
                        "endpoint_prediction": np.concatenate([part["endpoint_prediction"] for part in parts]),
                        "endpoint_linf_pixels": np.concatenate([part["endpoint_linf_pixels"] for part in parts]),
                        "target_loss": np.concatenate([part["target_loss"] for part in parts]),
                        "endpoint_features": np.concatenate([part["endpoint_features"] for part in parts]),
                    }
                    for position, sample_index in enumerate(indices):
                        attack_rows.append({
                            "model_group": model_group,
                            "seed": int(seed),
                            "trigger_source": trigger_sources[model_group],
                            "sample_index": int(sample_index),
                            "epsilon_pixels": float(eps_pixels),
                            "before_prediction": int(original_predictions[position]),
                            "after_prediction": int(endpoint_arrays[float(eps_pixels)]["endpoint_prediction"][position]),
                            "success": bool(endpoint_arrays[float(eps_pixels)]["success"][position]),
                            "pre_target": bool(pre_target[position]),
                            "actual_linf_pixels": float(endpoint_arrays[float(eps_pixels)]["endpoint_linf_pixels"][position]),
                            "targeted_loss": float(endpoint_arrays[float(eps_pixels)]["target_loss"][position]),
                            "target_resistance_margin": float(target_margins[position]),
                            "first_success_epsilon_pixels": None,
                            "censored": None,
                            "target": int(args.target),
                        })

                first_eps = np.full(len(indices), np.inf, dtype=np.float64)
                first_endpoint_features = endpoint_arrays[float(analysis_eps[-1])]["endpoint_features"].copy()
                first_endpoint_data = endpoint_arrays[float(analysis_eps[-1])]
                for eps_pixels in analysis_eps:
                    data_at_eps = endpoint_arrays[float(eps_pixels)]
                    newly = np.isinf(first_eps) & data_at_eps["success"]
                    first_eps[newly] = float(eps_pixels)
                    first_endpoint_features[newly] = data_at_eps["endpoint_features"][newly]
                    for position, sample_index in enumerate(indices):
                        if newly[position]:
                            first_endpoint_data["endpoint_prediction"][position] = data_at_eps["endpoint_prediction"][position]
                            first_endpoint_data["endpoint_linf_pixels"][position] = data_at_eps["endpoint_linf_pixels"][position]
                            first_endpoint_data["target_loss"][position] = data_at_eps["target_loss"][position]
                censored = np.isinf(first_eps)
                for row in attack_rows:
                    if row["model_group"] == model_group and int(row["seed"]) == int(seed) and int(row["sample_index"]) in indices:
                        position = indices.index(int(row["sample_index"]))
                        row["first_success_epsilon_pixels"] = None if censored[position] else float(first_eps[position])
                        row["censored"] = bool(censored[position])

                protocols: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [(
                    "first_success", first_endpoint_features,
                    first_endpoint_data["success"], first_eps,
                )]
                for eps_pixels in fixed_eps:
                    data_at_eps = endpoint_arrays[float(eps_pixels)]
                    protocols.append((
                        f"fixed_{eps_pixels:g}", data_at_eps["endpoint_features"],
                        data_at_eps["success"], np.full(len(indices), float(eps_pixels)),
                    ))

                for protocol, adversarial_features, success_values, protocol_eps in protocols:
                    adversarial_delta = adversarial_features - base_features
                    same_alignment = cosine_rows(adversarial_delta, trigger_delta)
                    global_direction_mask = (~pre_target) & np.isfinite(np.linalg.norm(trigger_delta, axis=1))
                    global_direction = trigger_delta[global_direction_mask].mean(axis=0) if global_direction_mask.any() else np.zeros(trigger_delta.shape[1])
                    global_alignment = cosine_rows(adversarial_delta, np.repeat(global_direction[None, :], len(indices), axis=0))
                    shuffled = shuffled_alignment(adversarial_delta, trigger_delta, seed=args.candidate_seed + seed, repeats=args.shuffle_repeats)
                    trigger_concentration = trigger_concentration_all
                    for position, sample_index in enumerate(indices):
                        sample_group = "pre_target" if pre_target[position] else ("success" if bool(success_values[position]) else "failure")
                        valid_adv = np.isfinite(np.linalg.norm(adversarial_delta[position])) and np.linalg.norm(adversarial_delta[position]) > 0
                        valid_trigger = np.isfinite(np.linalg.norm(trigger_delta[position])) and np.linalg.norm(trigger_delta[position]) > 0
                        eligible_adv = (not pre_target[position]) and valid_adv and valid_trigger
                        group_mask = (~pre_target) & (success_values.astype(bool) if sample_group == "success" else ~success_values.astype(bool))
                        adv_concentration = pairwise_concentration(adversarial_delta[group_mask]) if sample_group in {"success", "failure"} else pairwise_concentration(adversarial_delta[~pre_target])
                        alignment_rows.append({
                            "model_group": model_group,
                            "seed": int(seed),
                            "protocol": protocol,
                            "sample_index": int(sample_index),
                            "sample_group": sample_group,
                            "success": bool(success_values[position]),
                            "pre_target": bool(pre_target[position]),
                            "trigger_success": bool(trigger_success[position]),
                            "first_success_epsilon_pixels": None if not math.isfinite(float(first_eps[position])) else float(first_eps[position]),
                            "censored": bool(censored[position]),
                            "same_alignment": float(same_alignment[position]) if math.isfinite(float(same_alignment[position])) else None,
                            "shuffled_alignment": float(shuffled[position]) if math.isfinite(float(shuffled[position])) else None,
                            "same_minus_shuffled": float(same_alignment[position] - shuffled[position]) if math.isfinite(float(same_alignment[position])) and math.isfinite(float(shuffled[position])) else None,
                            "global_alignment": float(global_alignment[position]) if math.isfinite(float(global_alignment[position])) else None,
                            "trigger_feature_norm": float(np.linalg.norm(trigger_delta[position])),
                            "adversarial_feature_norm": float(np.linalg.norm(adversarial_delta[position])),
                            "target_resistance_margin": float(target_margins[position]),
                            "valid_for_alignment": bool(eligible_adv),
                            "trigger_concentration": trigger_concentration,
                            "trigger_concentration_success": trigger_concentration_success,
                            "adv_concentration": adv_concentration,
                        })
            finally:
                extractor.close()
        write_log(output, f"seed{seed}: completed Clean/BadNet feature analysis")

    metric_rows: list[dict] = []
    for model_group, _checkpoint_group in model_specs:
        for seed in clean_seeds:
            for protocol in ["first_success"] + [f"fixed_{eps:g}" for eps in fixed_eps]:
                subset = [row for row in alignment_rows if row["model_group"] == model_group and row["seed"] == seed and row["protocol"] == protocol]
                for sample_group in ("all", "success", "failure", "trigger_success"):
                    if sample_group == "all":
                        selected = [row for row in subset if not row["pre_target"]]
                    elif sample_group == "trigger_success":
                        selected = [row for row in subset if not row["pre_target"] and row["trigger_success"]]
                    else:
                        selected = [row for row in subset if row["sample_group"] == sample_group]
                    metric_rows.append({
                        "model_group": model_group,
                        "seed": int(seed),
                        "protocol": protocol,
                        "sample_group": sample_group,
                        "count": len(selected),
                        "success_rate": float(np.mean([row["success"] for row in selected])) if selected else None,
                        "trigger_success_rate": float(np.mean([row["trigger_success"] for row in selected])) if selected else None,
                        "alignment_mean": finite_stats(optional_float_array([row["same_alignment"] for row in selected]))["mean"] if selected else None,
                        "alignment_median": finite_stats(optional_float_array([row["same_alignment"] for row in selected]))["median"] if selected else None,
                        "shuffled_alignment_mean": finite_stats(optional_float_array([row["shuffled_alignment"] for row in selected]))["mean"] if selected else None,
                        "same_minus_shuffled_mean": finite_stats(optional_float_array([row["same_minus_shuffled"] for row in selected]))["mean"] if selected else None,
                        "global_alignment_mean": finite_stats(optional_float_array([row["global_alignment"] for row in selected]))["mean"] if selected else None,
                        "trigger_concentration": selected[0]["trigger_concentration"] if selected else None,
                        "trigger_concentration_success": selected[0]["trigger_concentration_success"] if selected else None,
                        "adv_concentration": selected[0]["adv_concentration"] if selected else None,
                        "censored_fraction": float(np.mean([row["censored"] for row in selected])) if selected else None,
                        "alignment_margin_spearman": spearman(optional_float_array([row["same_alignment"] for row in selected]), optional_float_array([row["target_resistance_margin"] for row in selected])) if selected else None,
                        "alignment_first_radius_spearman": spearman(optional_float_array([row["same_alignment"] for row in selected]), optional_float_array([row["first_success_epsilon_pixels"] for row in selected])) if selected else None,
                    })

    quality_rows = read_quality_report(args.quality_report)
    if not quality_rows:
        quality_rows = [{"group": group, "seed": seed, "status": "not_provided"} for _, group in model_specs for seed in clean_seeds]
    write_csv(output / "targeted_selection_records.csv", selection_rows)
    write_csv(output / "selected_targeted_robust_samples.csv", selected_rows)
    write_csv(output / "attack_records.csv", attack_rows)
    write_csv(output / "trigger_records.csv", trigger_rows)
    write_csv(output / "feature_alignment_records.csv", alignment_rows)
    write_csv(output / "group_metrics.csv", metric_rows)
    write_csv(output / "model_quality.csv", quality_rows)
    make_figures(output, alignment_rows)

    summary = {
        "protocol": "stage1d-targeted-robust-trigger-alignment-v1",
        "target": int(args.target),
        "clean_group": args.clean_group,
        "backdoor_groups": backdoor_groups,
        "clean_seeds": clean_seeds,
        "candidate_count": len(candidate_indices),
        "candidate_seed": int(args.candidate_seed),
        "top_coarse": int(args.top_coarse),
        "top_final": int(args.top_final),
        "selection_epsilon_pixels": list(selection_eps),
        "analysis_epsilon_pixels": list(analysis_eps),
        "fixed_report_epsilon_pixels": list(fixed_eps),
        "trigger_sources": trigger_sources,
        "blended_alpha": float(args.blended_alpha),
        "wanet_s": float(args.wanet_s),
        "wanet_grid_rescale": float(args.wanet_grid_rescale),
        "feature_layer": "model.avgpool",
        "feature_dimension": feature_dimension,
        "shuffle_repeats": int(args.shuffle_repeats),
        "selected_counts": {str(seed): len(indices) for seed, indices in selected_by_seed.items()},
        "group_metrics": metric_rows,
        "model_quality": quality_rows,
        "output_directory": str(output.resolve()),
    }
    write_json(output / "summary.json", summary)
    write_log(output, f"Stage 1D complete: {output.resolve()}")


if __name__ == "__main__":
    main()
