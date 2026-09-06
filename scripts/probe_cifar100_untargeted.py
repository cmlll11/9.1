"""Run the Stage 1C target-free robustness transfer experiment.

The experiment has two strictly separated parts.  Part A trains a target-free
Ridge Probe on CIFAR-100 train images with Clean seeds 3 and 4, then evaluates
Probe, top-1/top-2 margin, and a refined-PGD reference on CIFAR-100 test images
using Clean seeds 0--2.  Part B models the deployment setting: every model
selects its own samples from its own logits, and the selected images are then
attacked with untargeted PGD.  A fixed per-seed Random-100 set is included as
the no-selection baseline.

All images are raw tensors in [0, 1].  The loaded classifier wrapper applies
CIFAR-10 normalization internally.  CIFAR-100 labels are never used for
selection, Probe fitting, or attack success decisions.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar10_dataset, cifar100_dataset
from mdluap.probes import (
    UNTARGETED_FEATURE_NAMES,
    RidgeProbe,
    spearman_correlation,
    untargeted_logits_features,
    untargeted_margin,
)
from mdluap.targeted_pgd import UNTARGETED_EPSILON_GRID, untargeted_pgd
from pilot_common import batch_images, load_model, seed_everything, timestamp_run_dir, write_csv, write_json


EPSILON_PIXELS = tuple(float(value * 255.0) for value in UNTARGETED_EPSILON_GRID)
LOW_EPSILON_PIXELS = (0.25, 0.5, 0.75, 1.0)


def parse_ints(value: str) -> list[int]:
    """Parse a comma-separated integer list."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def checkpoint_path(model_root: Path, group: str, seed: int) -> Path:
    """Resolve the packaged BackdoorBench checkpoint path."""

    return model_root / group / f"seed{seed}" / "attack_result.pt"


def parse_args() -> argparse.Namespace:
    """Parse the fixed Stage 1C protocol and server paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output-root", default="results/stage1c_cifar100_untargeted")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--reference-clean-seeds", default="3,4")
    parser.add_argument("--target-clean-seeds", default="0,1,2")
    parser.add_argument("--backdoor-groups", default="badnet,lf,blended,wanet")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=2028)
    parser.add_argument("--top-coarse", type=int, default=300)
    parser.add_argument("--top-final", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--quality-report",
        default="",
        help="Optional existing model-gate JSON containing native trigger ASR values.",
    )
    return parser.parse_args()


def initial_predictions(model, dataset, indices: list[int], *, batch_size: int, device: torch.device) -> list[int]:
    """Return predictions in the exact order of ``indices``."""

    predictions: list[int] = []
    with torch.inference_mode():
        for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
            predictions.extend(model(images).argmax(dim=1).cpu().tolist())
    return [int(value) for value in predictions]


@torch.inference_mode()
def logits_for_indices(model, dataset, indices: list[int], *, batch_size: int, device: torch.device):
    """Return logits and dataset labels for indexed raw images."""

    logits_rows = []
    label_rows = []
    for _, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits_rows.append(model(images).detach().cpu())
        label_rows.append(labels.detach().cpu())
    return torch.cat(logits_rows), torch.cat(label_rows)


@torch.inference_mode()
def accuracy(model, dataset, indices: list[int], *, batch_size: int, device: torch.device) -> float:
    """Compute accuracy on a dataset subset; used only for model-quality logs."""

    correct = 0
    total = 0
    for _, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        correct += int(model(images).argmax(dim=1).eq(labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def fixed_pool_indices(length: int, count: int, seed: int) -> list[int]:
    """Draw a reproducible pool without using CIFAR-100 labels or predictions."""

    if count > length:
        raise ValueError(f"requested {count} samples from a dataset of length {length}")
    return np.random.default_rng(int(seed)).permutation(length)[: int(count)].astype(int).tolist()


def run_untargeted_grid(
    model,
    dataset,
    indices: list[int],
    *,
    model_name: str,
    steps: int,
    random_start: bool,
    restarts: int,
    batch_size: int,
    device: torch.device,
    phase: str,
) -> tuple[list[dict], dict[int, float], list[int]]:
    """Run untargeted PGD over the fixed grid and return long records.

    ``radius`` is the first epsilon *budget* at which a prediction changes.
    The attack is rerun independently for every epsilon, so raw per-budget
    success may be non-monotone.  ``cumulative_success`` is therefore also
    recorded and is the primary ASR statistic downstream.  A missing radius
    means the sample is right-censored beyond 32/255.
    """

    before = initial_predictions(model, dataset, indices, batch_size=batch_size, device=device)
    radius = {index: math.inf for index in indices}
    found = {index: False for index in indices}
    cumulative = {index: False for index in indices}
    records: list[dict] = []

    for epsilon_pixels in (0.0,) + EPSILON_PIXELS:
        epsilon = float(epsilon_pixels / 255.0)
        if epsilon == 0.0:
            raw_success = torch.zeros(len(indices), dtype=torch.bool)
            actual_norm = torch.zeros(len(indices), dtype=torch.float32)
            after = torch.tensor(before, dtype=torch.long)
        else:
            success_rows = []
            norm_rows = []
            prediction_rows = []
            for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
                result = untargeted_pgd(
                    model,
                    images,
                    epsilon=epsilon,
                    steps=steps,
                    alpha=epsilon / 10.0,
                    random_start=random_start,
                    restarts=restarts,
                )
                success_rows.append(result.success.detach().cpu())
                norm_rows.append(result.best_linf.detach().cpu())
                prediction_rows.append(result.best_prediction.detach().cpu())
            raw_success = torch.cat(success_rows)
            actual_norm = torch.cat(norm_rows)
            after = torch.cat(prediction_rows)

        for offset, sample_index in enumerate(indices):
            raw = bool(raw_success[offset])
            cumulative[sample_index] = cumulative[sample_index] or raw
            if epsilon > 0.0 and raw and not found[sample_index]:
                radius[sample_index] = float(epsilon_pixels)
                found[sample_index] = True
            records.append(
                {
                    "phase": phase,
                    "model_name": model_name,
                    "sample_index": int(sample_index),
                    "epsilon": epsilon,
                    "epsilon_pixels": float(epsilon_pixels),
                    "steps": int(steps),
                    "restarts": int(restarts),
                    "random_start": bool(random_start),
                    "before_prediction": int(before[offset]),
                    "after_prediction": int(after[offset]),
                    "success": raw,
                    "cumulative_success": bool(cumulative[sample_index]),
                    "actual_best_linf": float(actual_norm[offset]),
                }
            )

    for row in records:
        value = radius[int(row["sample_index"])]
        row["estimated_radius_pixels"] = "" if math.isinf(value) else float(value)
        row["censored"] = bool(math.isinf(value))
    return records, radius, before


def select_top(indices: list[int], scores: dict[int, float], count: int) -> list[int]:
    """Select the largest score values with stable sample-index tie breaking."""

    ordered = sorted(indices, key=lambda index: (-float(scores[index]), int(index)))
    return ordered[: int(count)]


def attach_selector_records(
    raw_records: list[dict],
    *,
    selectors_by_sample: dict[int, list[str]],
    analysis_part: str,
    target_clean_seed: int,
    model_group: str,
    model_seed: int,
) -> list[dict]:
    """Duplicate attack rows for selector membership and add comparison context."""

    output: list[dict] = []
    for row in raw_records:
        sample_index = int(row["sample_index"])
        for selector in selectors_by_sample.get(sample_index, []):
            output.append(
                {
                    **row,
                    "analysis_part": analysis_part,
                    "target_clean_seed": int(target_clean_seed),
                    "model_group": model_group,
                    "model_seed": int(model_seed),
                    "selector": selector,
                }
            )
    return output


def summarize_attack_records(rows: list[dict]) -> list[dict]:
    """Create cumulative ASR and finite-radius summaries from long records."""

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["analysis_part"],
                int(row["target_clean_seed"]),
                row["selector"],
                row["model_group"],
                int(row["model_seed"]),
                float(row["epsilon_pixels"]),
            )
        ].append(row)

    output: list[dict] = []
    for key, values in sorted(grouped.items()):
        part, target_seed, selector, group, model_seed, epsilon_pixels = key
        cumulative_success_count = sum(bool(row["cumulative_success"]) for row in values)
        raw_success_count = sum(bool(row["success"]) for row in values)
        output.append(
            {
                "metric": "cumulative_asr",
                "analysis_part": part,
                "target_clean_seed": target_seed,
                "selector": selector,
                "model_group": group,
                "model_seed": model_seed,
                "epsilon_pixels": epsilon_pixels,
                "n": len(values),
                "success_count": cumulative_success_count,
                "raw_success_count": raw_success_count,
                "asr": cumulative_success_count / len(values),
            }
        )

    sample_groups: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = (
            row["analysis_part"],
            int(row["target_clean_seed"]),
            row["selector"],
            row["model_group"],
            int(row["model_seed"]),
        )
        value = row["estimated_radius_pixels"]
        sample_groups[key][int(row["sample_index"])] = math.inf if value == "" else float(value)

    for key, values in sorted(sample_groups.items()):
        part, target_seed, selector, group, model_seed = key
        finite = [value for value in values.values() if math.isfinite(value)]
        output.append(
            {
                "metric": "radius",
                "analysis_part": part,
                "target_clean_seed": target_seed,
                "selector": selector,
                "model_group": group,
                "model_seed": model_seed,
                "epsilon_pixels": "",
                "n": len(values),
                "radius_n_finite": len(finite),
                "radius_censored_count": len(values) - len(finite),
                "radius_censored_fraction": (len(values) - len(finite)) / len(values),
                "radius_mean_finite_pixels": float(np.mean(finite)) if finite else "",
                "radius_median_finite_pixels": float(np.median(finite)) if finite else "",
                "radius_q25_finite_pixels": float(np.quantile(finite, 0.25)) if finite else "",
                "radius_q75_finite_pixels": float(np.quantile(finite, 0.75)) if finite else "",
            }
        )
    return output


def load_quality_report(path: str) -> dict[tuple[str, int], dict]:
    """Load optional native-trigger quality values from an existing JSON report."""

    if not path:
        return {}
    report_path = Path(path)
    if not report_path.is_file():
        return {}
    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    models = payload.get("models", payload)
    output: dict[tuple[str, int], dict] = {}
    if isinstance(models, dict):
        entries = models.items()
    elif isinstance(models, list):
        entries = ((str(index), item) for index, item in enumerate(models))
    else:
        return {}
    for key, value in entries:
        if not isinstance(value, dict):
            continue
        group = str(value.get("group", ""))
        seed_value = value.get("seed")
        if not group or seed_value is None:
            text_key = str(key)
            for candidate in ("badnet", "lf", "blended", "wanet", "clean_select_shared"):
                if text_key.startswith(candidate + "_seed"):
                    group = candidate
                    seed_value = text_key.rsplit("seed", 1)[-1]
                    break
        if not group or seed_value is None:
            continue
        native = value.get("backdoor_asr", value.get("native_trigger_asr", value.get("asr")))
        output[(group, int(seed_value))] = {"native_trigger_asr": native, "quality_source": str(report_path)}
    return output


def model_quality_row(model, group: str, seed: int, path: Path, cifar10_test, *, batch_size: int, device: torch.device, quality: dict) -> dict:
    """Record clean accuracy and optional native-trigger ASR for one checkpoint."""

    clean_accuracy = "" if cifar10_test is None else accuracy(
        model,
        cifar10_test,
        list(range(len(cifar10_test))),
        batch_size=batch_size,
        device=device,
    )
    extra = quality.get((group, int(seed)), {})
    return {
        "model_group": group,
        "model_seed": int(seed),
        "checkpoint": str(path),
        "clean_accuracy": clean_accuracy,
        "native_trigger_asr": extra.get("native_trigger_asr", ""),
        "quality_source": extra.get("quality_source", "not_supplied"),
    }


def write_yaml(path: Path, payload: dict) -> None:
    """Write a readable resolved configuration without requiring YAML at import time."""

    try:
        import yaml
    except ImportError:
        path.write_text(str(payload), encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def plot_results(output: Path, attack_rows: list[dict]) -> None:
    """Save compact ASR plots when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    grouped: dict[tuple, dict[float, float]] = defaultdict(dict)
    for row in summarize_attack_records(attack_rows):
        if row["metric"] != "cumulative_asr":
            continue
        key = (row["analysis_part"], row["model_group"], row["selector"], row["model_seed"])
        grouped[key][float(row["epsilon_pixels"])] = float(row["asr"])

    for part, filename in (("part_a", "part_a_asr.png"), ("part_b", "part_b_asr.png")):
        plt.figure(figsize=(8, 5))
        for (row_part, group, selector, seed), values in sorted(grouped.items()):
            if row_part != part:
                continue
            xs = sorted(values)
            plt.plot(xs, [values[x] for x in xs], marker="o", label=f"{group}-s{seed}-{selector}")
        plt.xlabel("epsilon (pixels / 255)")
        plt.ylabel("cumulative ASR")
        plt.title(part)
        plt.grid(alpha=0.25)
        if plt.gca().lines:
            plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(output / "figures" / filename, dpi=160)
        plt.close()


def make_config(args: argparse.Namespace) -> dict:
    """Return the fixed protocol configuration stored with every run."""

    return {
        "protocol": "cifar100-probe-untargeted-v1",
        "reference_clean_seeds": parse_ints(args.reference_clean_seeds),
        "target_clean_seeds": parse_ints(args.target_clean_seeds),
        "backdoor_groups": [item.strip() for item in args.backdoor_groups.split(",") if item.strip()],
        "train_count": int(args.train_count),
        "test_count": int(args.test_count),
        "split_seed": int(args.split_seed),
        "epsilon_pixels": list(EPSILON_PIXELS),
        "primary_low_epsilon_pixels": list(LOW_EPSILON_PIXELS),
        "coarse_pgd": {"steps": 30, "alpha_fraction": 0.1, "random_start": False, "restarts": 1},
        "refined_pgd": {"steps": 100, "alpha_fraction": 0.1, "random_start": True, "restarts": 3},
        "top_coarse": int(args.top_coarse),
        "top_final": int(args.top_final),
        "ridge_alpha": float(args.ridge_alpha),
        "batch_size": int(args.batch_size),
        "device": args.device,
        "target_free_features": list(UNTARGETED_FEATURE_NAMES),
    }


def main() -> None:
    """Execute Stage 1C and write a non-overwriting timestamped run directory."""

    args = parse_args()
    seed_everything(args.split_seed)
    config = make_config(args)
    output = timestamp_run_dir(args.output_root, "untargeted")
    write_yaml(output / "config.resolved.yaml", config)
    log_path = output / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    device = torch.device(args.device)
    model_root = Path(args.model_root)
    reference_seeds = parse_ints(args.reference_clean_seeds)
    target_seeds = parse_ints(args.target_clean_seeds)
    backdoor_groups = [item.strip() for item in args.backdoor_groups.split(",") if item.strip()]
    if set(reference_seeds) & set(target_seeds):
        raise ValueError("reference and target Clean seeds must be disjoint")

    train_dataset = cifar100_dataset(args.data_root, train=True)
    test_dataset = cifar100_dataset(args.data_root, train=False)
    # CIFAR-10 accuracy is useful quality context but is not required for the
    # OOD attack experiment.  Keep the run usable when only CIFAR-100 was
    # copied to the server; the missing value is saved as an empty CSV field.
    cifar10_root = Path(args.data_root) / "cifar10"
    cifar10_test = cifar10_dataset(args.data_root, train=False) if cifar10_root.is_dir() else None
    train_indices = fixed_pool_indices(len(train_dataset), args.train_count, args.split_seed)
    test_indices = fixed_pool_indices(len(test_dataset), args.test_count, args.split_seed + 1)
    random_indices_by_seed = {
        seed: fixed_pool_indices(len(test_indices), args.top_final, args.split_seed + 1000 + seed)
        for seed in target_seeds
    }
    random_samples_by_seed = {seed: [test_indices[position] for position in positions] for seed, positions in random_indices_by_seed.items()}

    split_rows = []
    split_rows.extend({"split": "probe_train", "pool_position": pos, "sample_index": index, "dataset": "CIFAR100_train"} for pos, index in enumerate(train_indices))
    split_rows.extend({"split": "target_test", "pool_position": pos, "sample_index": index, "dataset": "CIFAR100_test"} for pos, index in enumerate(test_indices))
    write_csv(output / "cifar100_split.csv", split_rows)

    model_paths: dict[str, str] = {}
    model_quality_rows: list[dict] = []
    quality_seen: set[tuple[str, int]] = set()
    quality_report = load_quality_report(args.quality_report)

    def record_quality(model, group: str, seed: int, path: Path) -> None:
        """Record each checkpoint once even when it is reused in Part A/B."""

        key = (group, int(seed))
        if key in quality_seen:
            return
        model_quality_rows.append(
            model_quality_row(
                model,
                group,
                seed,
                path,
                cifar10_test,
                batch_size=args.batch_size,
                device=device,
                quality=quality_report,
            )
        )
        quality_seen.add(key)

    reference_models = {}
    for seed in reference_seeds:
        path = checkpoint_path(model_root, args.clean_group, seed)
        model, _ = load_model(path, args.backdoorbench_root, device)
        reference_models[seed] = model
        model_paths[f"{args.clean_group}_seed{seed}"] = str(path)
        record_quality(model, args.clean_group, seed, path)

    reference_records: list[dict] = []
    reference_attack_records: list[dict] = []
    fit_features = []
    fit_labels = []
    log("Generating target-free Probe labels on CIFAR-100 train pool...")
    for seed, model in reference_models.items():
        logits, _ = logits_for_indices(model, train_dataset, train_indices, batch_size=args.batch_size, device=device)
        features = untargeted_logits_features(logits).numpy()
        margins = untargeted_margin(logits).numpy()
        attack_rows, radii, _ = run_untargeted_grid(
            model,
            train_dataset,
            train_indices,
            model_name=f"{args.clean_group}_seed{seed}",
            steps=30,
            random_start=False,
            restarts=1,
            batch_size=args.batch_size,
            device=device,
            phase="reference_probe_label",
        )
        reference_attack_records.extend([{**row, "reference_clean_seed": seed} for row in attack_rows])
        for position, index in enumerate(train_indices):
            radius = radii[index]
            row = {
                "reference_clean_seed": seed,
                "sample_index": int(index),
                "top1_top2_margin": float(margins[position]),
                "radius_pixels": "" if math.isinf(radius) else float(radius),
                "censored": bool(math.isinf(radius)),
            }
            row.update({f"feature_{name}": float(features[position, column]) for column, name in enumerate(UNTARGETED_FEATURE_NAMES)})
            reference_records.append(row)
            if math.isfinite(radius):
                fit_features.append(features[position])
                fit_labels.append(radius)

    if not fit_features:
        raise RuntimeError("no finite reference untargeted PGD labels are available for Probe fitting")
    probe = RidgeProbe.fit(
        np.asarray(fit_features),
        np.asarray(fit_labels),
        alpha=args.ridge_alpha,
        feature_names=UNTARGETED_FEATURE_NAMES,
    )
    probe.save(output / "probe_model.npz")
    write_csv(output / "reference_probe_records.csv", reference_records)
    write_csv(output / "reference_probe_attack_records.csv", reference_attack_records)
    log(f"Fitted target-free Probe on {len(fit_labels)} finite model-sample labels.")

    clean_selection_records: list[dict] = []
    paired_records: list[dict] = []
    pool_score_rows: list[dict] = []
    selector_rows: list[dict] = []
    ranking_rows: list[dict] = []
    all_attack_rows: list[dict] = []

    # Part A: target Clean models only.  No backdoor model participates in
    # selector construction or refined-reference ranking.
    for seed in target_seeds:
        clean_path = checkpoint_path(model_root, args.clean_group, seed)
        clean_model, _ = load_model(clean_path, args.backdoorbench_root, device)
        model_paths[f"{args.clean_group}_seed{seed}"] = str(clean_path)
        record_quality(clean_model, args.clean_group, seed, clean_path)
        log(f"Part A: scoring CIFAR-100 test pool with Clean seed{seed}...")
        logits, _ = logits_for_indices(clean_model, test_dataset, test_indices, batch_size=args.batch_size, device=device)
        features = untargeted_logits_features(logits).numpy()
        margins = untargeted_margin(logits).numpy()
        probe_scores = probe.predict(features)
        coarse_rows, coarse_radius, before_list = run_untargeted_grid(
            clean_model,
            test_dataset,
            test_indices,
            model_name=f"{args.clean_group}_seed{seed}",
            steps=30,
            random_start=False,
            restarts=1,
            batch_size=args.batch_size,
            device=device,
            phase="part_a_coarse",
        )
        before = {index: int(prediction) for index, prediction in zip(test_indices, before_list)}
        probe_by_index = {index: float(score) for index, score in zip(test_indices, probe_scores)}
        margin_by_index = {index: float(score) for index, score in zip(test_indices, margins)}
        coarse_by_index = {index: float(value) for index, value in coarse_radius.items()}
        for position, index in enumerate(test_indices):
            value = coarse_by_index[index]
            pool_score_rows.append(
                {
                    "analysis_part": "part_a",
                    "model_group": args.clean_group,
                    "model_seed": seed,
                    "sample_index": int(index),
                    "initial_prediction": before[index],
                    "probe_score": float(probe_scores[position]),
                    "target_free_margin": float(margins[position]),
                    "coarse_radius_pixels": "" if math.isinf(value) else value,
                    "coarse_censored": bool(math.isinf(value)),
                }
            )

        coarse_top = select_top(test_indices, coarse_by_index, args.top_coarse)
        probe_top = select_top(test_indices, probe_by_index, args.top_final)
        margin_top = select_top(test_indices, margin_by_index, args.top_final)
        # The strong pass on CoarseTop300 creates the reference and is reused
        # for every overlap with ProbeTop100/MarginTop100.
        strong_indices = sorted(set(coarse_top) | set(probe_top) | set(margin_top))
        log(f"Part A: strong PGD on {len(strong_indices)} unique samples (coarse Top-{len(coarse_top)} plus selector extras)...")
        strong_rows, strong_radius, _ = run_untargeted_grid(
            clean_model,
            test_dataset,
            strong_indices,
            model_name=f"{args.clean_group}_seed{seed}",
            steps=100,
            random_start=True,
            restarts=3,
            batch_size=args.batch_size,
            device=device,
            phase="part_a_strong",
        )
        refined_top = select_top(coarse_top, strong_radius, args.top_final)
        selected_sets = {
            "probe": probe_top,
            "target_margin": margin_top,
            "refined_pgd_reference": refined_top,
        }
        selectors_by_sample: dict[int, list[str]] = defaultdict(list)
        for selector, selected in selected_sets.items():
            for rank, index in enumerate(selected, start=1):
                selectors_by_sample[index].append(selector)
                strong_value = strong_radius[index]
                selector_rows.append(
                    {
                        "analysis_part": "part_a",
                        "selection_model_group": args.clean_group,
                        "selection_model_seed": seed,
                        "selector": selector,
                        "rank": rank,
                        "sample_index": int(index),
                        "probe_score": probe_by_index[index],
                        "target_free_margin": margin_by_index[index],
                        "coarse_radius_pixels": "" if math.isinf(coarse_by_index[index]) else coarse_by_index[index],
                        "strong_radius_pixels": "" if math.isinf(strong_value) else strong_value,
                    }
                )

        refined_finite = [index for index in coarse_top if math.isfinite(strong_radius[index])]
        ranking_rows.extend(
            {
                "analysis_part": "part_a",
                "target_clean_seed": seed,
                "metric": "spearman_vs_strong_radius_on_coarse_top",
                "selector": selector,
                "count": len(refined_finite),
                "spearman": spearman_correlation(
                    np.asarray([strong_radius[index] for index in refined_finite]),
                    np.asarray([scores[index] for index in refined_finite]),
                ) if len(refined_finite) >= 2 else None,
            }
            for selector, scores in (("probe", probe_by_index), ("target_margin", margin_by_index), ("coarse_pgd_reference", coarse_by_index))
        )

        clean_selection_records.extend(
            attach_selector_records(
                strong_rows,
                selectors_by_sample=selectors_by_sample,
                analysis_part="part_a",
                target_clean_seed=seed,
                model_group="clean",
                model_seed=seed,
            )
        )

    # Part B: load each model independently and select from that model's own
    # logits.  Random samples are fixed by seed and shared by all model groups
    # for that seed, making the Clean/Backdoor random comparison interpretable.
    for group, seeds in [(args.clean_group, target_seeds)] + [(group, target_seeds) for group in backdoor_groups]:
        for seed in seeds:
            path = checkpoint_path(model_root, group, seed)
            model_paths[f"{group}_seed{seed}"] = str(path)
            model, _ = load_model(path, args.backdoorbench_root, device)
            record_quality(model, group, seed, path)
            log(f"Part B: selecting and attacking {group}_seed{seed} independently...")
            logits, _ = logits_for_indices(model, test_dataset, test_indices, batch_size=args.batch_size, device=device)
            features = untargeted_logits_features(logits).numpy()
            margins = untargeted_margin(logits).numpy()
            probe_scores = probe.predict(features)
            probe_by_index = {index: float(score) for index, score in zip(test_indices, probe_scores)}
            margin_by_index = {index: float(score) for index, score in zip(test_indices, margins)}
            probe_top = select_top(test_indices, probe_by_index, args.top_final)
            margin_top = select_top(test_indices, margin_by_index, args.top_final)
            random_top = random_samples_by_seed[seed]
            selected_sets = {"probe": probe_top, "target_margin": margin_top, "random": random_top}
            selectors_by_sample: dict[int, list[str]] = defaultdict(list)
            for selector, selected in selected_sets.items():
                for rank, index in enumerate(selected, start=1):
                    selectors_by_sample[index].append(selector)
                    selector_rows.append(
                        {
                            "analysis_part": "part_b",
                            "selection_model_group": group,
                            "selection_model_seed": seed,
                            "selector": selector,
                            "rank": rank,
                            "sample_index": int(index),
                            "probe_score": probe_by_index[index],
                            "target_free_margin": margin_by_index[index],
                            "random_set_seed": args.split_seed + 1000 + seed if selector == "random" else "",
                        }
                    )
            selected_indices = sorted(selectors_by_sample)
            raw_rows, _, _ = run_untargeted_grid(
                model,
                test_dataset,
                selected_indices,
                model_name=f"{group}_seed{seed}",
                steps=100,
                random_start=True,
                restarts=3,
                batch_size=args.batch_size,
                device=device,
                phase="part_b_strong",
            )
            paired_records.extend(
                attach_selector_records(
                    raw_rows,
                    selectors_by_sample=selectors_by_sample,
                    analysis_part="part_b",
                    target_clean_seed=seed,
                    model_group=group,
                    model_seed=seed,
                )
            )
            for position, index in enumerate(test_indices):
                pool_score_rows.append(
                    {
                        "analysis_part": "part_b",
                        "model_group": group,
                        "model_seed": seed,
                        "sample_index": int(index),
                        "initial_prediction": int(logits[position].argmax().item()),
                        "probe_score": float(probe_scores[position]),
                        "target_free_margin": float(margins[position]),
                    }
                )

    all_attack_rows = clean_selection_records + paired_records
    group_metrics = summarize_attack_records(all_attack_rows)
    write_csv(output / "target_clean_pool_scores.csv", pool_score_rows)
    write_csv(output / "selector_sets.csv", selector_rows)
    write_csv(output / "clean_selection_attack_records.csv", clean_selection_records)
    write_csv(output / "paired_backdoor_attack_records.csv", paired_records)
    write_csv(output / "group_metrics.csv", group_metrics)
    write_csv(output / "ranking_metrics.csv", ranking_rows)
    write_csv(output / "model_quality.csv", model_quality_rows)

    # Model-level paired deltas.  Probe and margin selections are model
    # specific; random selection is the common per-seed baseline.
    metric_lookup = {
        (row["analysis_part"], row["target_clean_seed"], row["selector"], row["model_group"], row["model_seed"], row["epsilon_pixels"]): row
        for row in group_metrics
        if row["metric"] == "cumulative_asr"
    }
    delta_rows = []
    for seed in target_seeds:
        for selector in ("probe", "target_margin"):
            for group in backdoor_groups:
                for epsilon_pixels in LOW_EPSILON_PIXELS:
                    clean = metric_lookup.get(("part_b", seed, selector, args.clean_group, seed, epsilon_pixels), {})
                    bd = metric_lookup.get(("part_b", seed, selector, group, seed, epsilon_pixels), {})
                    random_clean = metric_lookup.get(("part_b", seed, "random", args.clean_group, seed, epsilon_pixels), {})
                    random_bd = metric_lookup.get(("part_b", seed, "random", group, seed, epsilon_pixels), {})
                    if clean and bd and random_clean and random_bd:
                        delta_probe = float(bd["asr"]) - float(clean["asr"])
                        delta_random = float(random_bd["asr"]) - float(random_clean["asr"])
                        delta_rows.append(
                            {
                                "target_seed": seed,
                                "backdoor_group": group,
                                "epsilon_pixels": epsilon_pixels,
                                "selector": selector,
                                "clean_asr": clean["asr"],
                                "backdoor_asr": bd["asr"],
                                "delta_asr_selector": delta_probe,
                                "random_clean_asr": random_clean["asr"],
                                "random_backdoor_asr": random_bd["asr"],
                                "delta_asr_random": delta_random,
                                "selector_minus_random_delta_asr": delta_probe - delta_random,
                            }
                        )
    write_csv(output / "paired_delta_metrics.csv", delta_rows)
    plot_results(output, all_attack_rows)

    summary = {
        **config,
        "model_paths": model_paths,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "random_samples_by_seed": random_samples_by_seed,
        "reference_finite_label_count": len(fit_labels),
        "reference_censored_label_count": len(reference_records) - len(fit_labels),
        "reference_censor_fraction": (len(reference_records) - len(fit_labels)) / max(len(reference_records), 1),
        "part_a": {
            "purpose": "Clean-only selector quality",
            "selectors": ["probe", "target_margin", "refined_pgd_reference"],
            "ranking_metrics": ranking_rows,
        },
        "part_b": {
            "purpose": "Independent deployment-style Clean/Backdoor comparison",
            "selectors": ["probe", "target_margin", "random"],
            "paired_delta_metrics": delta_rows,
        },
        "model_quality": model_quality_rows,
        "circularity_controls": {
            "probe_fit_models": reference_seeds,
            "probe_fit_data": "CIFAR100 train pool only",
            "target_clean_seeds_not_used_for_probe_fit": target_seeds,
            "target_model_pgd_not_used_for_probe_or_margin_selection": True,
            "refined_pgd_reference_used_only_in_part_a": True,
        },
    }
    write_json(output / "summary.json", summary)
    log(f"Untargeted transfer experiment complete: {output.resolve()}")


if __name__ == "__main__":
    main()
