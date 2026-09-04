"""Run Stage 1B within-model and leave-one-clean-model-out Probe tests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar10_dataset, cifar100_dataset
from mdluap.probes import RidgeProbe, logits_features, spearman_correlation, target_margin
from mdluap.targeted_pgd import EPSILON_GRID
from oracle_bridge_pilot import checkpoint_path, run_grid, ints
from pilot_common import batch_images, load_model, load_partition_indices, seed_everything, timestamp_run_dir, write_csv, write_json


def parse_args() -> argparse.Namespace:
    """Parse Stage 1B paths and fixed sample counts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--partition-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--stage1a-candidates", required=True)
    parser.add_argument("--output-root", default="results/stage1b_probe")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--clean-seeds", default="0,1,2")
    parser.add_argument("--backdoor-groups", default="blended,wanet")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--source-count", type=int, default=300)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--ood-count", type=int, default=200)
    parser.add_argument("--source-seed", type=int, default=2026)
    parser.add_argument("--ood-seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def select_source_indices(model, dataset, available: list[int], *, count: int, seed: int, target: int, batch_size: int, device: torch.device) -> tuple[list[int], list[int]]:
    """Select model-correct, non-target images from its known training set."""

    logits, labels = _logits_for_indices(model, dataset, available, batch_size=batch_size, device=device)
    predictions = logits.argmax(dim=1)
    eligible = [
        index for index, label, prediction in zip(available, labels.tolist(), predictions.tolist())
        if int(label) != int(target) and int(prediction) == int(label) and int(prediction) != int(target)
    ]
    order = np.random.default_rng(int(seed)).permutation(len(eligible)).tolist()
    shuffled = [eligible[position] for position in order]
    needed = int(count)
    if len(shuffled) < needed:
        raise RuntimeError(f"only {len(shuffled)} eligible training images found; need {needed}")
    return shuffled[:needed], shuffled[needed:]


@torch.inference_mode()
def _logits_for_indices(model, dataset, indices: list[int], *, batch_size: int, device: torch.device):
    """Compute logits while retaining labels for source-sample selection."""

    logits_rows = []
    label_rows = []
    for _, images, labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits_rows.append(model(images).cpu())
        label_rows.append(labels.cpu())
    return torch.cat(logits_rows), torch.cat(label_rows)


def finite_training_rows(record: dict) -> np.ndarray:
    """Return the rows whose coarse PGD radius was observed rather than censored."""

    return np.asarray([np.isfinite(value) for value in record["radii"]], dtype=bool)


def collect_clean_record(model, dataset, indices: list[int], *, seed: int, target: int, batch_size: int, device: torch.device) -> dict:
    """Extract logits, features and coarse PGD labels for one Clean model."""

    logits, labels = _logits_for_indices(model, dataset, indices, batch_size=batch_size, device=device)
    rows, radius, _ = run_grid(
        model, dataset, indices, model_name=f"clean_seed{seed}", target=target,
        steps=30, restarts=1, random_start=False, batch_size=batch_size, device=device, phase="probe_label",
    )
    radii = np.asarray([radius.get(index, float("inf")) for index in indices], dtype=np.float64)
    return {
        "seed": seed,
        "indices": indices,
        "labels": labels.numpy(),
        "logits": logits.numpy(),
        "features": logits_features(logits, target).numpy(),
        "target_margin": target_margin(logits, target).numpy(),
        "radii": radii,
        "attack_rows": rows,
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return finite-row Probe metrics in a JSON-friendly form."""

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    censored_count = int((~np.isfinite(actual)).sum())
    mask = np.isfinite(actual) & np.isfinite(predicted)
    finite_actual = actual[mask]
    finite_predicted = predicted[mask]
    return {
        "count": int(mask.sum()),
        "spearman": spearman_correlation(finite_actual, finite_predicted) if len(finite_actual) else None,
        "mae": float(np.mean(np.abs(finite_actual - finite_predicted))) if len(finite_actual) else None,
        "actual_median": float(np.median(finite_actual)) if len(finite_actual) else None,
        "predicted_median": float(np.median(finite_predicted)) if len(finite_predicted) else None,
        "censored_count": censored_count,
    }


def load_excluded_indices(path: str) -> set[int]:
    """Read Stage 1A candidate indices so Stage 1B uses a disjoint OOD pool."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {int(row["sample_index"]) for row in csv.DictReader(handle)}


def select_ood(dataset, *, count: int, seed: int, excluded: set[int]) -> list[int]:
    """Select a fixed OOD pool without reusing Stage 1A candidate indices."""

    order = np.random.default_rng(int(seed)).permutation(len(dataset)).tolist()
    selected = [int(index) for index in order if int(index) not in excluded][: int(count)]
    if len(selected) < int(count):
        raise RuntimeError("not enough disjoint CIFAR-100 OOD samples")
    return selected


def top_indices(indices: list[int], values: np.ndarray, predictions: list[int], target: int, count: int) -> list[int]:
    """Select largest scores after excluding images already predicted as target."""

    eligible = [(index, float(value)) for index, value, prediction in zip(indices, values, predictions) if prediction != int(target)]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [index for index, _ in eligible[: int(count)]]


def main() -> None:
    """Execute within-model, LOCO and OOD Probe evaluations."""

    args = parse_args()
    seed_everything(args.source_seed)
    device = torch.device(args.device)
    output = timestamp_run_dir(args.output_root, "stage1b")
    clean_seeds = ints(args.clean_seeds)
    backdoor_groups = [item.strip() for item in args.backdoor_groups.split(",") if item.strip()]
    train_dataset = cifar10_dataset(args.data_root, train=True)
    available = load_partition_indices(args.partition_root, "shared")

    records: dict[int, dict] = {}
    models: dict[int, object] = {}
    attack_rows: list[dict] = []
    source_rows: list[dict] = []
    within_rows: list[dict] = []
    for seed in clean_seeds:
        model_path = checkpoint_path(Path(args.model_root), args.clean_group, seed)
        model, _ = load_model(model_path, args.backdoorbench_root, device)
        models[seed] = model
        selected, validation = select_source_indices(
            model, train_dataset, available, count=args.source_count + args.validation_count,
            seed=args.source_seed + seed, target=args.target, batch_size=args.batch_size, device=device,
        )
        record = collect_clean_record(model, train_dataset, selected[: args.source_count], seed=seed, target=args.target, batch_size=args.batch_size, device=device)
        validation_record = collect_clean_record(model, train_dataset, selected[args.source_count :], seed=seed, target=args.target, batch_size=args.batch_size, device=device)
        record["validation"] = validation_record
        records[seed] = record
        attack_rows.extend([{**row, "clean_seed": seed} for row in record["attack_rows"]])
        attack_rows.extend([{**row, "clean_seed": seed, "split": "validation"} for row in validation_record["attack_rows"]])
        finite = finite_training_rows(record)
        if finite.sum() < max(10, args.source_count // 2):
            raise RuntimeError(f"too few finite PGD labels for clean seed {seed}")
        probe = RidgeProbe.fit(record["features"][finite], record["radii"][finite])
        val_features = validation_record["features"]
        val_radii = validation_record["radii"]
        within_pred = probe.predict(val_features)
        within_rows.append({"protocol": "within_model", "clean_seed": seed, "probe": regression_metrics(val_radii, within_pred)})

    loco_rows: list[dict] = []
    for held_out in clean_seeds:
        train_parts = []
        held_validation_indices = set(records[held_out]["validation"]["indices"])
        for seed in clean_seeds:
            if seed == held_out:
                continue
            source = records[seed]
            mask = finite_training_rows(source) & np.asarray([index not in held_validation_indices for index in source["indices"]])
            if mask.any():
                train_parts.append((source["features"][mask], source["radii"][mask]))
        train_features = np.concatenate([part[0] for part in train_parts], axis=0)
        train_radii = np.concatenate([part[1] for part in train_parts], axis=0)
        probe = RidgeProbe.fit(train_features, train_radii)
        validation = records[held_out]["validation"]
        prediction = probe.predict(validation["features"])
        margin_prediction = validation["target_margin"]
        loco_rows.append(
            {
                "protocol": "leave_one_clean_model_out",
                "held_out_seed": held_out,
                "train_seeds": [seed for seed in clean_seeds if seed != held_out],
                "probe": regression_metrics(validation["radii"], prediction),
                "target_margin_baseline": regression_metrics(validation["radii"], margin_prediction),
            }
        )

    excluded = load_excluded_indices(args.stage1a_candidates)
    ood_dataset = cifar100_dataset(args.data_root, train=False)
    ood_indices = select_ood(ood_dataset, count=args.ood_count, seed=args.ood_seed, excluded=excluded)
    ood_selection_rows: list[dict] = []
    ood_attack_rows: list[dict] = []
    for held_out in clean_seeds:
        model = models[held_out]
        logits, _ = _logits_for_indices(model, ood_dataset, ood_indices, batch_size=args.batch_size, device=device)
        features = logits_features(logits, args.target).numpy()
        margins = target_margin(logits, args.target).numpy()
        predictions = logits.argmax(dim=1).tolist()
        train_features = []
        train_radii = []
        for seed in clean_seeds:
            if seed == held_out:
                continue
            source = records[seed]
            mask = finite_training_rows(source)
            train_features.append(source["features"][mask])
            train_radii.append(source["radii"][mask])
        probe = RidgeProbe.fit(np.concatenate(train_features), np.concatenate(train_radii))
        probe_scores = probe.predict(features)
        probe_selected = top_indices(ood_indices, probe_scores, predictions, args.target, 30)
        margin_selected = top_indices(ood_indices, margins, predictions, args.target, 30)
        for selector, selected in (("probe", probe_selected), ("target_margin", margin_selected)):
            for rank, sample_index in enumerate(selected, start=1):
                position = ood_indices.index(sample_index)
                ood_selection_rows.append({
                    "held_out_clean_seed": held_out,
                    "selector": selector,
                    "rank": rank,
                    "sample_index": sample_index,
                    "probe_score": float(probe_scores[position]),
                    "target_margin": float(margins[position]),
                    "initial_prediction": int(predictions[position]),
                })
            eval_specs = [("clean", model)]
            for group in backdoor_groups:
                path = checkpoint_path(Path(args.model_root), group, held_out)
                eval_specs.append((group, load_model(path, args.backdoorbench_root, device)[0]))
            for group, eval_model in eval_specs:
                evaluated, _, _ = run_grid(
                    eval_model, ood_dataset, selected, model_name=f"{group}_seed{held_out}", target=args.target,
                    steps=100, restarts=3, random_start=True, batch_size=args.batch_size, device=device, phase="ood_evaluation",
                )
                ood_attack_rows.extend([{**row, "held_out_clean_seed": held_out, "selector": selector, "model_group": group} for row in evaluated])

    write_csv(output / "probe_label_attack.csv", attack_rows)
    write_csv(output / "within_model_results.csv", within_rows)
    write_csv(output / "loco_results.csv", loco_rows)
    write_csv(output / "ood_selection.csv", ood_selection_rows)
    write_csv(output / "ood_attack_results.csv", ood_attack_rows)
    summary = {
        "protocol": "stage1b-probe-loco-v1",
        "clean_seeds": clean_seeds,
        "source_count": args.source_count,
        "validation_count": args.validation_count,
        "ood_count": args.ood_count,
        "within_model": within_rows,
        "leave_one_clean_model_out": loco_rows,
        "ood_selection_counts": {selector: sum(row["selector"] == selector for row in ood_selection_rows) for selector in ("probe", "target_margin")},
    }
    write_json(output / "summary.json", summary)
    print(f"Stage 1B complete: {output.resolve()}")


if __name__ == "__main__":
    main()
