"""Train a robustness Probe on CIFAR-100 and transfer it to CIFAR-10 models.

The experiment is deliberately split into two analyses:

1. Clean-only selector comparison: Probe, target margin, and a refined-PGD
   reference select CIFAR-100 samples for three target Clean models.
2. Same-seed backdoor comparison: the fixed selections are evaluated on the
   paired Clean, BadNet, LF, Blended, and WaNet checkpoints.

The CIFAR-100 train and test splits are disjoint.  Clean seed3/4 provide the
Probe training records; Clean seed0/1/2 are never used to fit the Probe.
Images are raw tensors in [0, 1], and model wrappers apply CIFAR-10
normalization internally.
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

from mdluap.data import cifar100_dataset
from mdluap.probes import FEATURE_NAMES, RidgeProbe, logits_features, spearman_correlation, target_margin
from mdluap.targeted_pgd import FINE_EPSILON_GRID, targeted_pgd
from pilot_common import batch_images, load_model, seed_everything, timestamp_run_dir, write_csv, write_json


def parse_ints(value: str) -> list[int]:
    """Parse a comma-separated integer list."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def checkpoint_path(model_root: Path, group: str, seed: int) -> Path:
    """Return the packaged BackdoorBench checkpoint path."""

    return model_root / group / f"seed{seed}" / "attack_result.pt"


def parse_args() -> argparse.Namespace:
    """Parse the fixed transfer protocol and server paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output-root", default="results/stage1b_cifar100_transfer")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--reference-clean-seeds", default="3,4")
    parser.add_argument("--target-clean-seeds", default="0,1,2")
    parser.add_argument("--backdoor-groups", default="badnet,lf,blended,wanet")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=2028)
    parser.add_argument("--top-coarse", type=int, default=300)
    parser.add_argument("--top-final", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def initial_predictions(model, dataset, indices: list[int], *, batch_size: int, device: torch.device) -> list[int]:
    """Return model predictions for indexed raw images in stable order."""

    predictions: list[int] = []
    with torch.inference_mode():
        for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
            predictions.extend(model(images).argmax(dim=1).cpu().tolist())
    return [int(value) for value in predictions]


@torch.inference_mode()
def logits_for_indices(model, dataset, indices: list[int], *, batch_size: int, device: torch.device):
    """Return logits and dataset labels for indexed images."""

    logits_rows = []
    label_rows = []
    for _, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits_rows.append(model(images).detach().cpu())
        label_rows.append(labels.detach().cpu())
    return torch.cat(logits_rows), torch.cat(label_rows)


def common_non_target_pool(
    models: list[object],
    dataset,
    *,
    count: int,
    seed: int,
    target: int,
    batch_size: int,
    device: torch.device,
) -> list[int]:
    """Select a common pool whose reference models all avoid target class.

    CIFAR-100 labels are intentionally ignored.  The only eligibility rule is
    that every supplied CIFAR-10 model predicts a class other than the target,
    which removes trivial ASR@0 samples while preserving a common pool.
    """

    order = np.random.default_rng(int(seed)).permutation(len(dataset)).tolist()
    predictions = []
    for model in models:
        predictions.append(initial_predictions(model, dataset, order, batch_size=batch_size, device=device))
    eligible = [
        int(index)
        for position, index in enumerate(order)
        if all(prediction[position] != int(target) for prediction in predictions)
    ]
    if len(eligible) < int(count):
        raise RuntimeError(f"only {len(eligible)} common non-target samples are available; need {count}")
    return eligible[: int(count)]


def run_fine_grid(
    model,
    dataset,
    indices: list[int],
    *,
    model_name: str,
    target: int,
    steps: int,
    random_start: bool,
    restarts: int,
    batch_size: int,
    device: torch.device,
    phase: str,
) -> tuple[list[dict], dict[int, float], dict[int, bool], list[int]]:
    """Run the fine PGD grid and return long records plus first-success budgets.

    The ``radius`` dictionary stores the first successful grid budget in raw
    image units.  The attack record additionally stores the actual Linf norm
    found by PGD.  A missing radius means right-censored beyond 32/255.
    """

    before = initial_predictions(model, dataset, indices, batch_size=batch_size, device=device)
    radius = {index: 0.0 for index, prediction in zip(indices, before) if prediction == int(target)}
    found = {index: prediction == int(target) for index, prediction in zip(indices, before)}
    records: list[dict] = []

    for epsilon in (0.0,) + tuple(FINE_EPSILON_GRID):
        if epsilon == 0.0:
            successes = torch.tensor([prediction == int(target) for prediction in before], dtype=torch.bool)
            actual_norm = torch.zeros(len(indices), dtype=torch.float32)
            after = torch.tensor(before, dtype=torch.long)
        else:
            success_rows = []
            norm_rows = []
            prediction_rows = []
            for batch_indices, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
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
                success_rows.append(result.success.detach().cpu())
                norm_rows.append(result.best_linf.detach().cpu())
                prediction_rows.append(result.best_prediction.detach().cpu())
            successes = torch.cat(success_rows)
            actual_norm = torch.cat(norm_rows)
            after = torch.cat(prediction_rows)

        for offset, sample_index in enumerate(indices):
            success = bool(successes[offset])
            if success and not found[sample_index]:
                radius[sample_index] = float(epsilon)
                found[sample_index] = True
            records.append(
                {
                    "phase": phase,
                    "model_name": model_name,
                    "sample_index": int(sample_index),
                    "epsilon": float(epsilon),
                    "epsilon_pixels": float(epsilon * 255.0),
                    "steps": int(steps),
                    "restarts": int(restarts),
                    "random_start": bool(random_start),
                    "before_prediction": int(before[offset]),
                    "after_prediction": int(after[offset]),
                    "success": success,
                    "actual_best_linf": float(actual_norm[offset]),
                    "target": int(target),
                }
            )
    return records, radius, found, before


def select_top(indices: list[int], values: dict[int, float], before: dict[int, int], *, target: int, count: int) -> list[int]:
    """Select largest scores, placing right-censored samples first."""

    eligible = [index for index in indices if before[index] != int(target)]
    eligible.sort(key=lambda index: (-float(values.get(index, math.inf)), int(index)))
    return eligible[: int(count)]


def expand_records(
    raw_records: list[dict],
    *,
    selectors_by_sample: dict[int, list[str]],
    analysis_part: str,
    target_clean_seed: int,
    model_group: str,
    model_seed: int,
    radius: dict[int, float],
) -> list[dict]:
    """Attach selector membership and final sample radius to attack records."""

    output: list[dict] = []
    for row in raw_records:
        sample_index = int(row["sample_index"])
        for selector in selectors_by_sample.get(sample_index, []):
            value = float(radius.get(sample_index, math.inf))
            output.append(
                {
                    **row,
                    "analysis_part": analysis_part,
                    "target_clean_seed": int(target_clean_seed),
                    "model_group": model_group,
                    "model_seed": int(model_seed),
                    "selector": selector,
                    "estimated_radius_pixels": value * 255.0 if math.isfinite(value) else "",
                    "censored": not math.isfinite(value),
                }
            )
    return output


def radius_metrics(rows: list[dict]) -> list[dict]:
    """Summarize ASR curves and first-success radii by experiment group."""

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
        part, target_seed, selector, group, model_seed, epsilon = key
        success_count = sum(bool(row["success"]) for row in values)
        output.append(
            {
                "metric": "asr",
                "analysis_part": part,
                "target_clean_seed": target_seed,
                "selector": selector,
                "model_group": group,
                "model_seed": model_seed,
                "epsilon_pixels": epsilon,
                "n": len(values),
                "success_count": success_count,
                "asr": success_count / len(values),
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
        sample_groups[key][int(row["sample_index"])] = float(row["estimated_radius_pixels"]) if row["estimated_radius_pixels"] != "" else math.inf

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
                "success_count": len(finite),
                "asr": "",
                "radius_n_finite": len(finite),
                "radius_censored_count": len(values) - len(finite),
                "radius_mean_finite_pixels": float(np.mean(finite)) if finite else "",
                "radius_median_finite_pixels": float(np.median(finite)) if finite else "",
                "radius_q25_finite_pixels": float(np.quantile(finite, 0.25)) if finite else "",
                "radius_q75_finite_pixels": float(np.quantile(finite, 0.75)) if finite else "",
            }
        )
    return output


def make_config(args: argparse.Namespace) -> dict:
    """Return the resolved protocol configuration for YAML/JSON output."""

    return {
        "protocol": "cifar100-probe-transfer-v1",
        "target": int(args.target),
        "reference_clean_seeds": parse_ints(args.reference_clean_seeds),
        "target_clean_seeds": parse_ints(args.target_clean_seeds),
        "backdoor_groups": [item.strip() for item in args.backdoor_groups.split(",") if item.strip()],
        "train_count": int(args.train_count),
        "test_count": int(args.test_count),
        "split_seed": int(args.split_seed),
        "epsilon_pixels": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0],
        "coarse_pgd": {"steps": 30, "restarts": 1, "random_start": False},
        "refined_pgd": {"steps": 100, "restarts": 3, "random_start": True},
        "top_coarse": int(args.top_coarse),
        "top_final": int(args.top_final),
        "ridge_alpha": float(args.ridge_alpha),
        "batch_size": int(args.batch_size),
        "device": args.device,
    }


def write_yaml(path: Path, payload: dict) -> None:
    """Write resolved configuration using PyYAML when available."""

    try:
        import yaml
    except ImportError:
        path.write_text(str(payload), encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    """Execute both transfer analyses and write an independent run directory."""

    args = parse_args()
    seed_everything(args.split_seed)
    config = make_config(args)
    output = timestamp_run_dir(args.output_root, "transfer")
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
    reference_models = {}
    target_models = {}
    model_paths = {}

    for seed in reference_seeds + target_seeds:
        path = checkpoint_path(model_root, args.clean_group, seed)
        model, metadata = load_model(path, args.backdoorbench_root, device)
        model_paths[f"clean_seed{seed}"] = str(path)
        if seed in reference_seeds:
            reference_models[seed] = model
        else:
            target_models[seed] = model

    log("Selecting CIFAR-100 train pool using reference Clean models...")
    train_indices = common_non_target_pool(
        list(reference_models.values()),
        train_dataset,
        count=args.train_count,
        seed=args.split_seed,
        target=args.target,
        batch_size=args.batch_size,
        device=device,
    )
    log("Selecting CIFAR-100 test pool using target Clean models...")
    test_indices = common_non_target_pool(
        list(target_models.values()),
        test_dataset,
        count=args.test_count,
        seed=args.split_seed + 1,
        target=args.target,
        batch_size=args.batch_size,
        device=device,
    )
    split_rows = []
    for position, index in enumerate(train_indices):
        split_rows.append({"split": "probe_train", "pool_position": position, "sample_index": index, "dataset": "CIFAR100_train"})
    for position, index in enumerate(test_indices):
        split_rows.append({"split": "target_test", "pool_position": position, "sample_index": index, "dataset": "CIFAR100_test"})
    write_csv(output / "cifar100_split.csv", split_rows)

    reference_records = []
    reference_attack_records = []
    fit_features = []
    fit_labels = []
    for seed, model in reference_models.items():
        log(f"Generating fine-grid Probe labels with Clean seed{seed}...")
        logits, _ = logits_for_indices(model, train_dataset, train_indices, batch_size=args.batch_size, device=device)
        features = logits_features(logits, args.target).numpy()
        margins = target_margin(logits, args.target).numpy()
        attack_rows, radii, found, _ = run_fine_grid(
            model,
            train_dataset,
            train_indices,
            model_name=f"clean_seed{seed}",
            target=args.target,
            steps=30,
            random_start=False,
            restarts=1,
            batch_size=args.batch_size,
            device=device,
            phase="reference_probe_label",
        )
        reference_attack_records.extend([{**row, "reference_clean_seed": seed} for row in attack_rows])
        for position, index in enumerate(train_indices):
            value = float(radii.get(index, math.inf))
            row = {
                "reference_clean_seed": seed,
                "sample_index": int(index),
                "target_margin": float(margins[position]),
                "radius_pixels": value * 255.0 if math.isfinite(value) else "",
                "censored": not math.isfinite(value),
            }
            row.update({f"feature_{name}": float(features[position, column]) for column, name in enumerate(FEATURE_NAMES)})
            reference_records.append(row)
            if math.isfinite(value):
                fit_features.append(features[position])
                fit_labels.append(value)

    if not fit_features:
        raise RuntimeError("no finite reference PGD labels are available for Probe fitting")
    probe = RidgeProbe.fit(np.asarray(fit_features), np.asarray(fit_labels), alpha=args.ridge_alpha)
    probe.save(output / "probe_model.npz")
    write_csv(output / "reference_probe_records.csv", reference_records)
    write_csv(output / "reference_probe_attack_records.csv", reference_attack_records)
    log(f"Fitted Probe on {len(fit_labels)} finite model-sample records.")

    clean_attack_records = []
    paired_attack_records = []
    pool_score_rows = []
    selector_rows = []
    ranking_summary = []

    for seed, clean_model in target_models.items():
        log(f"Evaluating selectors on target Clean seed{seed}...")
        logits, _ = logits_for_indices(clean_model, test_dataset, test_indices, batch_size=args.batch_size, device=device)
        features = logits_features(logits, args.target).numpy()
        margins = target_margin(logits, args.target).numpy()
        probe_scores = probe.predict(features)
        coarse_rows, coarse_radius, coarse_found, before_list = run_fine_grid(
            clean_model,
            test_dataset,
            test_indices,
            model_name=f"clean_seed{seed}",
            target=args.target,
            steps=30,
            random_start=False,
            restarts=1,
            batch_size=args.batch_size,
            device=device,
            phase="target_clean_coarse",
        )
        before = {index: prediction for index, prediction in zip(test_indices, before_list)}
        probe_by_index = {index: float(score) for index, score in zip(test_indices, probe_scores)}
        margin_by_index = {index: float(score) for index, score in zip(test_indices, margins)}
        for position, index in enumerate(test_indices):
            pool_score_rows.append(
                {
                    "target_clean_seed": seed,
                    "sample_index": int(index),
                    "initial_prediction": before[index],
                    "probe_score": float(probe_scores[position]),
                    "target_margin": float(margins[position]),
                    "coarse_radius_pixels": float(coarse_radius[index] * 255.0) if math.isfinite(coarse_radius.get(index, math.inf)) else "",
                    "coarse_censored": not math.isfinite(coarse_radius.get(index, math.inf)),
                }
            )

        coarse_top = select_top(test_indices, coarse_radius, before, target=args.target, count=args.top_coarse)
        log(f"  refining coarse-PGD Top-{len(coarse_top)} with 100 steps x 3 restarts...")
        refined_rows, refined_radius, refined_found, refined_before_list = run_fine_grid(
            clean_model,
            test_dataset,
            coarse_top,
            model_name=f"clean_seed{seed}",
            target=args.target,
            steps=100,
            random_start=True,
            restarts=3,
            batch_size=args.batch_size,
            device=device,
            phase="target_clean_refined_reference",
        )
        refined_before = {index: prediction for index, prediction in zip(coarse_top, refined_before_list)}
        probe_top = select_top(test_indices, probe_by_index, before, target=args.target, count=args.top_final)
        margin_top = select_top(test_indices, margin_by_index, before, target=args.target, count=args.top_final)
        refined_top = select_top(coarse_top, refined_radius, refined_before, target=args.target, count=args.top_final)
        selected_sets = {
            "probe": probe_top,
            "target_margin": margin_top,
            "refined_pgd_reference": refined_top,
        }
        selectors_by_sample: dict[int, list[str]] = defaultdict(list)
        for selector, selected in selected_sets.items():
            for rank, index in enumerate(selected, start=1):
                selectors_by_sample[index].append(selector)
                selector_rows.append(
                    {
                        "target_clean_seed": seed,
                        "selector": selector,
                        "rank": rank,
                        "sample_index": int(index),
                        "probe_score": probe_by_index[index],
                        "target_margin": margin_by_index[index],
                        "coarse_radius_pixels": float(coarse_radius[index] * 255.0) if math.isfinite(coarse_radius.get(index, math.inf)) else "",
                        "refined_radius_pixels": float(refined_radius[index] * 255.0) if math.isfinite(refined_radius.get(index, math.inf)) else "",
                    }
                )

        refined_finite = {index: value for index, value in refined_radius.items() if math.isfinite(value)}
        for selector_name, selector_values in (("probe", probe_by_index), ("target_margin", margin_by_index), ("coarse_pgd_reference", coarse_radius)):
            common = [index for index in coarse_top if index in refined_finite and math.isfinite(selector_values.get(index, math.inf))]
            ranking_summary.append(
                {
                    "target_clean_seed": seed,
                    "metric": "spearman_vs_refined_radius_on_coarse_top",
                    "selector": selector_name,
                    "count": len(common),
                    "spearman": spearman_correlation(
                        np.asarray([refined_radius[index] for index in common]),
                        np.asarray([selector_values[index] for index in common]),
                    ) if len(common) >= 2 else None,
                }
            )

        union_indices = sorted(selectors_by_sample)
        clean_raw, clean_radius, _, _ = run_fine_grid(
            clean_model,
            test_dataset,
            union_indices,
            model_name=f"clean_seed{seed}",
            target=args.target,
            steps=100,
            random_start=True,
            restarts=3,
            batch_size=args.batch_size,
            device=device,
            phase="clean_selection_refined_evaluation",
        )
        clean_attack_records.extend(
            expand_records(
                clean_raw,
                selectors_by_sample=selectors_by_sample,
                analysis_part="clean_selection",
                target_clean_seed=seed,
                model_group="clean",
                model_seed=seed,
                radius=clean_radius,
            )
        )

        for group in backdoor_groups:
            path = checkpoint_path(model_root, group, seed)
            model_paths[f"{group}_seed{seed}"] = str(path)
            backdoor_model, _ = load_model(path, args.backdoorbench_root, device)
            log(f"  evaluating paired {group}_seed{seed}...")
            raw, radius, _, _ = run_fine_grid(
                backdoor_model,
                test_dataset,
                union_indices,
                model_name=f"{group}_seed{seed}",
                target=args.target,
                steps=100,
                random_start=True,
                restarts=3,
                batch_size=args.batch_size,
                device=device,
                phase="paired_backdoor_evaluation",
            )
            paired_attack_records.extend(
                expand_records(
                    raw,
                    selectors_by_sample=selectors_by_sample,
                    analysis_part="paired_backdoor",
                    target_clean_seed=seed,
                    model_group=group,
                    model_seed=seed,
                    radius=radius,
                )
            )

    write_csv(output / "target_clean_pool_scores.csv", pool_score_rows)
    write_csv(output / "selector_sets.csv", selector_rows)
    write_csv(output / "clean_selection_attack_records.csv", clean_attack_records)
    write_csv(output / "paired_backdoor_attack_records.csv", paired_attack_records)
    all_attack_records = clean_attack_records + paired_attack_records
    write_csv(output / "group_metrics.csv", radius_metrics(all_attack_records))
    write_csv(output / "ranking_metrics.csv", ranking_summary)

    summary = {
        **config,
        "model_paths": model_paths,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "reference_finite_label_count": len(fit_labels),
        "reference_censored_label_count": len(reference_records) - len(fit_labels),
        "selector_counts": {
            "probe": len([row for row in selector_rows if row["selector"] == "probe"]),
            "target_margin": len([row for row in selector_rows if row["selector"] == "target_margin"]),
            "refined_pgd_reference": len([row for row in selector_rows if row["selector"] == "refined_pgd_reference"]),
        },
        "ranking_metrics": ranking_summary,
    }
    write_json(output / "summary.json", summary)
    log(f"Transfer experiment complete: {output.resolve()}")


if __name__ == "__main__":
    main()
