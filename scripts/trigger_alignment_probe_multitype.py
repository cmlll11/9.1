"""Stage 1D fast validation with a CIFAR-100 Ridge Probe.

Clean seeds 1--3 train a target-0 robustness Probe on CIFAR-100 train
images.  Clean seed 0 uses that Probe to select one shared CIFAR-100 test
Top-100 set.  The set is then evaluated on Clean0 and one seed-0 model for
each requested backdoor family.  Every family uses its own official test-time
trigger adapter; missing trigger state disables only alignment, never the
targeted-PGD attack records.
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
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar100_dataset
from mdluap.official_triggers import TriggerAdapter, build_trigger_adapters
from mdluap.probes import FEATURE_NAMES, RidgeProbe, logits_features, spearman_correlation
from mdluap.targeted_pgd import targeted_pgd, targeted_pgd_endpoint
from mdluap.models import load_backdoor_toolbox_resnet18
from pilot_common import batch_images, load_model, timestamp_run_dir, write_csv, write_json


REFERENCE_SEEDS = (1, 2, 3)
BACKDOOR_GROUPS = ("badnet", "blended", "wanet", "ssba", "inputaware", "adaptive_blend")
PROBE_EPS_PIXELS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
ANALYSIS_EPS_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
FIXED_REPORT_EPS_PIXELS = (1.0, 1.5, 2.0)


class AvgPoolFeatureExtractor:
    """Capture the 512-dimensional PreActResNet18 avgpool representation."""

    def __init__(self, classifier: nn.Module):
        backbone = getattr(classifier, "model", None)
        layer = getattr(backbone, "avgpool", None)
        if layer is None:
            raise AttributeError("checkpoint model has no model.avgpool layer")
        self.output: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._hook)
        self.classifier = classifier

    def _hook(self, _module, _inputs, output):
        self.output = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(images)
        if self.output is None:
            raise RuntimeError("avgpool hook did not capture output")
        return logits.detach(), self.output.flatten(1).detach()

    def close(self) -> None:
        self.handle.remove()


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or tuple(sorted(values)) != values or any(value <= 0 for value in values):
        raise ValueError("epsilon list must be ascending positive values")
    return values


def checkpoint_path(model_root: Path, group: str, seed: int) -> Path:
    if group == "clean_select_shared":
        clean_path = model_root / group / f"seed{seed}" / "clean_model.pth"
        if clean_path.is_file():
            return clean_path
    aliases = {
        "inputaware": ("inputaware", "input_aware", "input-aware"),
        "adaptive_blend": ("adaptive_blend", "adaptive-blend", "adaptiveblend", "adap_blend"),
    }
    for candidate in aliases.get(group, (group,)):
        path = model_root / candidate / f"seed{seed}" / "attack_result.pt"
        if path.is_file():
            return path
    return model_root / group / f"seed{seed}" / "attack_result.pt"


def load_stage_model(
    model_root: Path,
    group: str,
    seed: int,
    *,
    backdoorbench_root: Path,
    device: torch.device,
    adaptive_blend_root: Path | None,
    adaptive_blend_model_path: Path | None,
):
    """Load either a BackdoorBench model or the official Adaptive-Blend model."""

    if group == "adaptive_blend":
        path = adaptive_blend_model_path or model_root / group / f"seed{seed}" / "official_model.pt"
        if adaptive_blend_root is None:
            raise ValueError("--adaptive-blend-root is required for Adaptive-Blend")
        return load_backdoor_toolbox_resnet18(
            str(path), backdoor_toolbox_root=str(adaptive_blend_root), device=device
        )
    return load_model(checkpoint_path(model_root, group, seed), str(backdoorbench_root), device)


def write_log(output: Path, message: str) -> None:
    line = message.rstrip()
    print(line, flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_quality_report(path: str | None) -> list[dict]:
    """Read the existing model-gate JSON without recomputing trigger ASR."""

    if not path:
        return []
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(report, dict) and isinstance(report.get("rows"), list):
        return report["rows"]
    if isinstance(report, dict) and isinstance(report.get("model_quality"), list):
        return report["model_quality"]
    if isinstance(report, list):
        return report
    return []


def finite_array(values: Iterable[object]) -> np.ndarray:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=np.float64)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = (left[valid] * right[valid]).sum(axis=1) / denominator[valid]
    return result


def pairwise_concentration(vectors: np.ndarray) -> float | None:
    if vectors.shape[0] < 2:
        return None
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    matrix = normalized @ normalized.T
    upper = matrix[np.triu_indices(vectors.shape[0], k=1)]
    upper = upper[np.isfinite(upper)]
    return float(upper.mean()) if upper.size else None


def shuffled_alignment(adv: np.ndarray, trigger: np.ndarray, *, seed: int, repeats: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(repeats)):
        permutation = rng.permutation(len(trigger))
        if len(trigger) > 1:
            fixed = np.arange(len(trigger))
            while np.any(permutation == fixed):
                permutation = rng.permutation(len(trigger))
        values.append(cosine_rows(adv, trigger[permutation]))
    return np.nanmean(np.stack(values), axis=0) if values else np.full(len(adv), np.nan)


def make_figures(output: Path, alignment_rows: list[dict]) -> None:
    """Write compact alignment figures when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups: dict[str, list[float]] = {}
    for row in alignment_rows:
        if row["protocol"] != "first_success" or not row["alignment_available"]:
            continue
        value = row.get("same_alignment")
        if value is None or not math.isfinite(float(value)):
            continue
        key = f"{row['model_alias']}:{row['trigger_type']}:{row['sample_group']}"
        groups.setdefault(key, []).append(float(value))
    if groups:
        labels = list(groups)
        plt.figure(figsize=(max(10, len(labels) * 0.55), 5))
        plt.boxplot([groups[label] for label in labels], labels=labels, showfliers=False)
        plt.ylabel("same-trigger feature alignment")
        plt.xticks(rotation=65, ha="right")
        plt.tight_layout()
        plt.savefig(output / "figures" / "alignment_boxplot.png", dpi=160)
        plt.close()

    concentration = {}
    for row in alignment_rows:
        if row["protocol"] == "first_success" and row["sample_group"] == "success" and row.get("adv_concentration") is not None:
            concentration[f"{row['model_alias']}:{row['trigger_type']}"] = float(row["adv_concentration"])
    if concentration:
        labels = list(concentration)
        plt.figure(figsize=(max(8, len(labels) * 0.8), 5))
        plt.bar(np.arange(len(labels)), [concentration[label] for label in labels])
        plt.ylabel("adversarial direction concentration")
        plt.xticks(np.arange(len(labels)), labels, rotation=65, ha="right")
        plt.tight_layout()
        plt.savefig(output / "figures" / "direction_concentration.png", dpi=160)
        plt.close()


@torch.inference_mode()
def logits_and_features(model: nn.Module, dataset, indices: list[int], *, batch_size: int, device: torch.device):
    logits_parts, feature_parts, predictions = [], [], []
    for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits = model(images).detach()
        logits_parts.append(logits.cpu().numpy())
        feature_parts.append(logits_features(logits, target=0).cpu().numpy())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return np.concatenate(logits_parts), np.concatenate(feature_parts), np.asarray(predictions, dtype=np.int64)


def targeted_grid_labels(
    model: nn.Module,
    dataset,
    indices: list[int],
    *,
    epsilon_pixels: tuple[float, ...],
    steps: int,
    batch_size: int,
    device: torch.device,
    model_group: str,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Return first-success pixel radii and long coarse-PGD records."""

    _, _, before_predictions = logits_and_features(model, dataset, indices, batch_size=batch_size, device=device)
    radius = np.full(len(indices), np.inf, dtype=np.float64)
    records: list[dict] = []
    for epsilon_pixels in epsilon_pixels:
        successes, predictions, norms = [], [], []
        for batch_indices, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
            result = targeted_pgd(
                model,
                images,
                target=0,
                epsilon=float(epsilon_pixels) / 255.0,
                steps=int(steps),
                alpha=float(epsilon_pixels) / 255.0 / 10.0,
                random_start=False,
                restarts=1,
            )
            successes.extend(result.success.cpu().tolist())
            predictions.extend(result.best_prediction.cpu().tolist())
            norms.extend((result.best_linf.cpu().numpy() * 255.0).tolist())
        for position, sample_index in enumerate(indices):
            success = bool(successes[position])
            if success and not math.isfinite(radius[position]):
                radius[position] = float(epsilon_pixels)
            records.append({
                "model_group": model_group,
                "seed": int(seed),
                "sample_index": int(sample_index),
                "epsilon_pixels": float(epsilon_pixels),
                "success": success,
                "before_prediction": int(before_predictions[position]),
                "after_prediction": int(predictions[position]),
                "actual_linf_pixels": float(norms[position]) if math.isfinite(norms[position]) else None,
                "target": 0,
                "steps": int(steps),
                "random_start": False,
                "restarts": 1,
            })
    return radius, records


def select_pool(dataset, count: int, seed: int) -> list[int]:
    order = np.random.default_rng(int(seed)).permutation(len(dataset))
    return [int(value) for value in order[: int(count)]]


def update_first_success(endpoint_by_eps: dict[float, dict], epsilon_pixels: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, dict]:
    count = len(next(iter(endpoint_by_eps.values()))["success"])
    first_eps = np.full(count, np.inf, dtype=np.float64)
    first_features = endpoint_by_eps[float(epsilon_pixels[-1])]["features"].copy()
    first_data = {key: value.copy() for key, value in endpoint_by_eps[float(epsilon_pixels[-1])].items() if key != "features"}
    for epsilon in epsilon_pixels:
        current = endpoint_by_eps[float(epsilon)]
        newly = np.isinf(first_eps) & current["success"]
        first_eps[newly] = float(epsilon)
        first_features[newly] = current["features"][newly]
        for key in first_data:
            first_data[key][newly] = current[key][newly]
    return first_eps, first_features, first_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--adaptive-blend-root", default=None)
    parser.add_argument("--adaptive-blend-model-path", default=None)
    parser.add_argument("--record-root", default=None)
    parser.add_argument("--output-root", default="results/stage1d_trigger_alignment")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--backdoor-groups", default=",".join(BACKDOOR_GROUPS))
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--probe-pool-count", type=int, default=1000)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--top-final", type=int, default=100)
    parser.add_argument("--candidate-seed", type=int, default=2031)
    parser.add_argument("--probe-eps-pixels", default=",".join(map(str, PROBE_EPS_PIXELS)))
    parser.add_argument("--analysis-eps-pixels", default=",".join(map(str, ANALYSIS_EPS_PIXELS)))
    parser.add_argument("--fixed-report-eps-pixels", default=",".join(map(str, FIXED_REPORT_EPS_PIXELS)))
    parser.add_argument("--probe-steps", type=int, default=30)
    parser.add_argument("--analysis-steps", type=int, default=100)
    parser.add_argument("--analysis-restarts", type=int, default=3)
    parser.add_argument("--probe-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--badnet-trigger-path", default=None)
    parser.add_argument("--blended-trigger-path", default=None)
    parser.add_argument("--wanet-state-path", default=None)
    parser.add_argument("--ssba-test-path", default=None)
    parser.add_argument("--inputaware-state-path", default=None)
    parser.add_argument("--adaptive-blend-trigger-path", default=None)
    parser.add_argument("--blended-alpha", type=float, default=0.2)
    parser.add_argument("--adaptive-blend-alpha", type=float, default=0.2)
    parser.add_argument("--wanet-s", type=float, default=0.5)
    parser.add_argument("--wanet-grid-rescale", type=float, default=1.0)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.target) != 0:
        raise ValueError("Stage 1D is fixed to CIFAR-10 target label 0")
    backdoor_groups = tuple(item.strip() for item in args.backdoor_groups.split(",") if item.strip())
    unknown = set(backdoor_groups) - set(BACKDOOR_GROUPS)
    if unknown:
        raise ValueError(f"unsupported backdoor groups: {sorted(unknown)}")
    probe_eps = parse_float_list(args.probe_eps_pixels)
    analysis_eps = parse_float_list(args.analysis_eps_pixels)
    fixed_eps = parse_float_list(args.fixed_report_eps_pixels)
    if not set(fixed_eps).issubset(set(analysis_eps)):
        raise ValueError("fixed report epsilon values must be in analysis epsilon values")

    output = timestamp_run_dir(args.output_root, "stage1d")
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    model_root = Path(args.model_root)
    backdoorbench_root = Path(args.backdoorbench_root)
    record_root = Path(args.record_root) if args.record_root else None
    adaptive_blend_root = Path(args.adaptive_blend_root) if args.adaptive_blend_root else None
    adaptive_blend_model_path = Path(args.adaptive_blend_model_path) if args.adaptive_blend_model_path else None
    train_data = cifar100_dataset(data_root, train=True)
    test_data = cifar100_dataset(data_root, train=False)
    train_indices = select_pool(train_data, args.probe_pool_count, args.candidate_seed + 11)
    test_indices = select_pool(test_data, args.candidate_count, args.candidate_seed + 12)
    write_log(output, f"Stage 1D Probe multitype started: {output.resolve()}")
    write_log(output, f"target=0 reference_clean_seeds={REFERENCE_SEEDS} test_models=seed0 groups={backdoor_groups}")

    write_csv(output / "candidate_pool.csv", [
        {"split": "cifar100_train", "pool_position": position, "sample_index": index}
        for position, index in enumerate(train_indices)
    ] + [
        {"split": "cifar100_test", "pool_position": position, "sample_index": index}
        for position, index in enumerate(test_indices)
    ])

    probe_features, probe_labels, probe_records = [], [], []
    for seed in REFERENCE_SEEDS:
        model, _ = load_model(checkpoint_path(model_root, args.clean_group, seed), str(backdoorbench_root), device)
        _, features, before_predictions = logits_and_features(model, train_data, train_indices, batch_size=args.batch_size, device=device)
        radius, radius_records = targeted_grid_labels(
            model, train_data, train_indices, epsilon_pixels=probe_eps, steps=args.probe_steps,
            batch_size=args.batch_size, device=device, model_group=args.clean_group, seed=seed,
        )
        for position, sample_index in enumerate(train_indices):
            label = None if not math.isfinite(radius[position]) else float(radius[position])
            row = {
                "clean_seed": int(seed), "sample_index": int(sample_index),
                "original_prediction": int(before_predictions[position]),
                "target": 0, "targeted_radius_pixels": label,
                "censored": not math.isfinite(radius[position]),
            }
            row.update({name: float(features[position, column]) for column, name in enumerate(FEATURE_NAMES)})
            probe_records.append(row)
        finite = np.isfinite(radius)
        probe_features.append(features[finite])
        probe_labels.append(radius[finite])
        write_log(output, f"probe seed{seed}: finite_labels={int(finite.sum())} censored={int((~finite).sum())}")

    probe_features_array = np.concatenate(probe_features, axis=0)
    probe_labels_array = np.concatenate(probe_labels, axis=0)
    probe = RidgeProbe.fit(
        probe_features_array,
        probe_labels_array,
        alpha=float(args.probe_alpha),
        feature_names=FEATURE_NAMES,
    )
    write_csv(output / "probe_training_records.csv", probe_records)
    write_json(output / "probe_parameters.json", {
        "feature_names": list(probe.feature_names),
        "feature_mean": probe.feature_mean.tolist(),
        "feature_std": probe.feature_std.tolist(),
        "weight": probe.weight.tolist(),
        "bias": float(probe.bias),
        "alpha": float(probe.alpha),
        "reference_clean_seeds": list(REFERENCE_SEEDS),
        "target": 0,
    })

    clean0, _ = load_stage_model(
        model_root, args.clean_group, 0, backdoorbench_root=backdoorbench_root,
        device=device, adaptive_blend_root=None, adaptive_blend_model_path=None,
    )
    test_logits, test_features, test_predictions = logits_and_features(clean0, test_data, test_indices, batch_size=args.batch_size, device=device)
    predicted_radius = probe.predict(test_features)
    eligible = test_predictions != 0
    ranked_positions = [position for position in np.argsort(-predicted_radius, kind="mergesort") if eligible[position]]
    selected_positions = ranked_positions[: int(args.top_final)]
    selected_indices = [test_indices[position] for position in selected_positions]
    selection_rows = []
    for rank, position in enumerate(selected_positions, start=1):
        selection_rows.append({
            "rank": rank, "sample_index": int(test_indices[position]),
            "original_prediction": int(test_predictions[position]),
            "probe_predicted_radius_pixels": float(predicted_radius[position]),
            "target": 0,
        })
    write_csv(output / "probe_selection_scores.csv", [
        {
            "sample_index": int(index), "original_prediction": int(test_predictions[position]),
            "probe_predicted_radius_pixels": float(predicted_radius[position]),
            "eligible": bool(eligible[position]), "target": 0,
        }
        for position, index in enumerate(test_indices)
    ])
    write_csv(output / "selected_targeted_robust_samples.csv", selection_rows)
    write_log(output, f"Clean0 Probe Top-{len(selected_indices)} selected from {len(test_indices)} CIFAR-100 test images")

    explicit = {
        "badnet": Path(args.badnet_trigger_path) if args.badnet_trigger_path else None,
        "blended": Path(args.blended_trigger_path) if args.blended_trigger_path else None,
        "wanet": Path(args.wanet_state_path) if args.wanet_state_path else None,
        "ssba": Path(args.ssba_test_path) if args.ssba_test_path else None,
        "inputaware": Path(args.inputaware_state_path) if args.inputaware_state_path else None,
        "adaptive_blend": Path(args.adaptive_blend_trigger_path) if args.adaptive_blend_trigger_path else None,
    }
    adapters = build_trigger_adapters(
        backdoorbench_root=backdoorbench_root, model_root=model_root, record_root=record_root,
        device=device, wanet_s=args.wanet_s, wanet_grid_rescale=args.wanet_grid_rescale,
        blended_alpha=args.blended_alpha, adaptive_blend_alpha=args.adaptive_blend_alpha,
        explicit=explicit,
    )

    attack_rows, trigger_rows, alignment_rows = [], [], []
    model_cache = {"clean": clean0}
    model_group_paths = {"clean": checkpoint_path(model_root, args.clean_group, 0)}
    for group in backdoor_groups:
        model_cache[group], _ = load_stage_model(
            model_root, group, 0, backdoorbench_root=backdoorbench_root,
            device=device, adaptive_blend_root=adaptive_blend_root,
            adaptive_blend_model_path=adaptive_blend_model_path,
        )
        model_group_paths[group] = (
            adaptive_blend_model_path or model_root / group / "seed0" / "official_model.pt"
            if group == "adaptive_blend"
            else checkpoint_path(model_root, group, 0)
        )

    for trigger_type in backdoor_groups:
        adapter = adapters[trigger_type]
        for model_group in ("clean",) + backdoor_groups:
            if model_group != "clean" and model_group != trigger_type:
                continue
            model = model_cache[model_group]
            extractor = AvgPoolFeatureExtractor(model)
            try:
                base_features_parts, base_logits_parts = [], []
                trigger_features_parts, trigger_logits_parts = [], []
                endpoint_by_eps = {float(eps): [] for eps in analysis_eps}
                for batch_indices, images, _ in batch_images(test_data, selected_indices, batch_size=args.batch_size, device=device):
                    base_logits, base_features = extractor(images)
                    base_logits_parts.append(base_logits.cpu().numpy())
                    base_features_parts.append(base_features.cpu().numpy())
                    if adapter.status.available:
                        triggered = adapter.apply(images, sample_indices=batch_indices, split="cifar100_test")
                        trigger_logits, trigger_features = extractor(triggered)
                        trigger_logits_parts.append(trigger_logits.cpu().numpy())
                        trigger_features_parts.append(trigger_features.cpu().numpy())
                    for eps_pixels in analysis_eps:
                        endpoint = targeted_pgd_endpoint(
                            model, images, target=0, epsilon=float(eps_pixels) / 255.0,
                            steps=args.analysis_steps, alpha=float(eps_pixels) / 255.0 / 10.0,
                            random_start=True, restarts=args.analysis_restarts,
                        )
                        _, endpoint_features = extractor(endpoint.endpoint)
                        endpoint_by_eps[float(eps_pixels)].append({
                            "success": endpoint.success.cpu().numpy(),
                            "prediction": endpoint.endpoint_prediction.cpu().numpy(),
                            "linf": endpoint.endpoint_linf.cpu().numpy() * 255.0,
                            "loss": endpoint.target_loss.cpu().numpy(),
                            "features": endpoint_features.cpu().numpy(),
                        })

                base_features = np.concatenate(base_features_parts)
                base_logits = np.concatenate(base_logits_parts)
                base_predictions = base_logits.argmax(axis=1)
                trigger_available = adapter.status.available
                if trigger_available:
                    trigger_features = np.concatenate(trigger_features_parts)
                    trigger_logits = np.concatenate(trigger_logits_parts)
                    trigger_predictions = trigger_logits.argmax(axis=1)
                    trigger_delta = trigger_features - base_features
                    trigger_success = trigger_predictions == 0
                    trigger_concentration = pairwise_concentration(trigger_delta[base_predictions != 0])
                else:
                    trigger_features = trigger_logits = trigger_delta = None
                    trigger_predictions = np.full(len(selected_indices), -1, dtype=np.int64)
                    trigger_success = np.zeros(len(selected_indices), dtype=bool)
                    trigger_concentration = None

                for position, sample_index in enumerate(selected_indices):
                    trigger_rows.append({
                        "model_group": model_group_paths[model_group].parent.parent.name,
                        "model_alias": model_group,
                        "seed": 0,
                        "trigger_type": trigger_type,
                        "sample_index": int(sample_index),
                        "trigger_source": adapter.status.source,
                        "trigger_available": trigger_available,
                        "alignment_available": trigger_available,
                        "alignment_unavailable_reason": adapter.status.reason if not trigger_available else None,
                        "original_prediction": int(base_predictions[position]),
                        "trigger_prediction": int(trigger_predictions[position]),
                        "trigger_success": bool(trigger_success[position]) if trigger_available else None,
                        "target": 0,
                    })

                endpoint_arrays = {}
                for eps_pixels in analysis_eps:
                    parts = endpoint_by_eps[float(eps_pixels)]
                    endpoint_arrays[float(eps_pixels)] = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
                first_eps, first_features, first_data = update_first_success(endpoint_arrays, analysis_eps)
                censored = np.isinf(first_eps)

                for eps_pixels in analysis_eps:
                    data_at_eps = endpoint_arrays[float(eps_pixels)]
                    for position, sample_index in enumerate(selected_indices):
                        attack_rows.append({
                            "model_group": model_group_paths[model_group].parent.parent.name,
                            "model_alias": model_group, "seed": 0, "trigger_type": trigger_type,
                            "sample_index": int(sample_index), "epsilon_pixels": float(eps_pixels),
                            "before_prediction": int(base_predictions[position]),
                            "after_prediction": int(data_at_eps["prediction"][position]),
                            "success": bool(data_at_eps["success"][position]),
                            "first_success_epsilon_pixels": None if censored[position] else float(first_eps[position]),
                            "censored": bool(censored[position]),
                            "actual_linf_pixels": float(data_at_eps["linf"][position]),
                            "targeted_loss": float(data_at_eps["loss"][position]),
                            "target": 0,
                        })

                protocols = [("first_success", first_features, ~censored)]
                protocols.extend((f"fixed_{eps:g}", endpoint_arrays[float(eps)]["features"], endpoint_arrays[float(eps)]["success"]) for eps in fixed_eps)
                target_margins = np.max(np.where(np.arange(base_logits.shape[1])[None, :] == 0, -np.inf, base_logits), axis=1) - base_logits[:, 0]
                for protocol, adversarial_features, success_values in protocols:
                    adversarial_delta = adversarial_features - base_features
                    if trigger_available:
                        same = cosine_rows(adversarial_delta, trigger_delta)
                        shuffled = shuffled_alignment(adversarial_delta, trigger_delta, seed=2031, repeats=args.shuffle_repeats)
                    else:
                        same = shuffled = np.full(len(selected_indices), np.nan)
                    for position, sample_index in enumerate(selected_indices):
                        pre_target = bool(base_predictions[position] == 0)
                        sample_group = "pre_target" if pre_target else ("success" if bool(success_values[position]) else "failure")
                        group_mask = (base_predictions != 0) & (success_values.astype(bool) if sample_group == "success" else ~success_values.astype(bool))
                        adv_concentration = pairwise_concentration(adversarial_delta[group_mask]) if sample_group in {"success", "failure"} else pairwise_concentration(adversarial_delta[base_predictions != 0])
                        alignment_rows.append({
                            "model_group": model_group_paths[model_group].parent.parent.name,
                            "model_alias": model_group, "seed": 0, "trigger_type": trigger_type,
                            "protocol": protocol, "sample_index": int(sample_index),
                            "sample_group": sample_group, "success": bool(success_values[position]),
                            "pre_target": pre_target, "trigger_success": bool(trigger_success[position]) if trigger_available else None,
                            "trigger_available": trigger_available, "alignment_available": trigger_available,
                            "same_alignment": float(same[position]) if math.isfinite(float(same[position])) else None,
                            "shuffled_alignment": float(shuffled[position]) if math.isfinite(float(shuffled[position])) else None,
                            "same_minus_shuffled": float(same[position] - shuffled[position]) if math.isfinite(float(same[position])) and math.isfinite(float(shuffled[position])) else None,
                            "trigger_feature_norm": float(np.linalg.norm(trigger_delta[position])) if trigger_available else None,
                            "adversarial_feature_norm": float(np.linalg.norm(adversarial_delta[position])),
                            "target_resistance_margin": float(target_margins[position]),
                            "first_success_epsilon_pixels": None if censored[position] else float(first_eps[position]),
                            "censored": bool(censored[position]), "trigger_concentration": trigger_concentration,
                            "adv_concentration": adv_concentration,
                        })
            finally:
                extractor.close()
        write_log(output, f"completed trigger_type={trigger_type} adapter_available={adapter.status.available}")

    metric_rows = []
    for trigger_type in backdoor_groups:
        for model_alias in ("clean", trigger_type):
            model_rows = [row for row in alignment_rows if row["trigger_type"] == trigger_type and row["model_alias"] == model_alias]
            for protocol in ["first_success"] + [f"fixed_{eps:g}" for eps in fixed_eps]:
                subset = [row for row in model_rows if row["protocol"] == protocol]
                for sample_group in ("all", "success", "failure", "trigger_success"):
                    if sample_group == "all":
                        selected = [row for row in subset if not row["pre_target"]]
                    elif sample_group == "trigger_success":
                        selected = [row for row in subset if not row["pre_target"] and row["trigger_success"] is True]
                    else:
                        selected = [row for row in subset if row["sample_group"] == sample_group]
                    metric_rows.append({
                        "model_alias": model_alias, "trigger_type": trigger_type, "seed": 0,
                        "protocol": protocol, "sample_group": sample_group, "count": len(selected),
                        "success_rate": float(np.mean([row["success"] for row in selected])) if selected else None,
                        "trigger_success_rate": float(np.mean([row["trigger_success"] for row in selected])) if selected and all(row["trigger_success"] is not None for row in selected) else None,
                        "alignment_mean": float(np.nanmean(finite_array(row["same_alignment"] for row in selected))) if finite_array(row["same_alignment"] for row in selected).size else None,
                        "shuffled_alignment_mean": float(np.nanmean(finite_array(row["shuffled_alignment"] for row in selected))) if finite_array(row["shuffled_alignment"] for row in selected).size else None,
                        "same_minus_shuffled_mean": float(np.nanmean(finite_array(row["same_minus_shuffled"] for row in selected))) if finite_array(row["same_minus_shuffled"] for row in selected).size else None,
                        "trigger_concentration": selected[0]["trigger_concentration"] if selected else None,
                        "adv_concentration": selected[0]["adv_concentration"] if selected else None,
                        "alignment_available": bool(selected and any(row["alignment_available"] for row in selected)),
                        "censored_fraction": float(np.mean([row["censored"] for row in selected])) if selected else None,
                    })

    quality_rows = read_quality_report(args.quality_report) if args.quality_report else []
    if not quality_rows:
        quality_rows = [{"group": args.clean_group, "seed": 0, "status": "not_provided"}] + [
            {"group": group, "seed": 0, "status": "not_provided"} for group in backdoor_groups
        ]
    write_csv(output / "attack_records.csv", attack_rows)
    write_csv(output / "trigger_records.csv", trigger_rows)
    write_csv(output / "feature_alignment_records.csv", alignment_rows)
    write_csv(output / "group_metrics.csv", metric_rows)
    write_csv(output / "model_quality.csv", quality_rows)
    make_figures(output, alignment_rows)
    try:
        import yaml
        (output / "config.resolved.yaml").write_text(yaml.safe_dump(vars(args), sort_keys=False), encoding="utf-8")
    except ImportError:
        write_json(output / "config.resolved.yaml", vars(args))
    summary = {
        "protocol": "stage1d-cifar100-probe-multitype-v1",
        "target": 0,
        "probe_reference_clean_seeds": list(REFERENCE_SEEDS),
        "test_clean_seed": 0,
        "backdoor_groups": list(backdoor_groups),
        "probe_train_count": len(train_indices),
        "probe_test_count": len(test_indices),
        "selected_count": len(selected_indices),
        "selected_indices": selected_indices,
        "trigger_status": {group: vars(adapter.status) for group, adapter in adapters.items()},
        "model_quality": quality_rows,
        "output_directory": str(output.resolve()),
    }
    write_json(output / "summary.json", summary)
    write_log(output, f"Stage 1D complete: {output.resolve()}")


if __name__ == "__main__":
    main()
