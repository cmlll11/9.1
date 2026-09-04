"""Run the minimal Stage 1A PGD-OOD bridge experiment.

The script first discovers target-0 robust CIFAR-100 images with a Clean
model, then evaluates exactly those images on the paired Clean, Blended and
WaNet models.  It intentionally does not train a Probe or use random controls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar100_dataset
from mdluap.targeted_pgd import EPSILON_GRID, targeted_pgd
from pilot_common import batch_images, load_model, seed_everything, timestamp_run_dir, write_csv, write_json


def parse_args() -> argparse.Namespace:
    """Parse the fixed pilot protocol and server-specific paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output-root", default="results/stage1a_bridge")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--clean-seeds", default="0,1")
    parser.add_argument("--backdoor-groups", default="blended,wanet")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=200)
    parser.add_argument("--candidate-seed", type=int, default=2026)
    parser.add_argument("--top-coarse", type=int, default=40)
    parser.add_argument("--top-final", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def ints(value: str) -> list[int]:
    """Parse a comma-separated integer list."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def checkpoint_path(model_root: Path, group: str, seed: int) -> Path:
    """Resolve the repository's packaged BackdoorBench checkpoint layout."""

    return model_root / group / f"seed{seed}" / "attack_result.pt"


def initial_predictions(model, dataset, indices: list[int], *, batch_size: int, device: torch.device) -> list[int]:
    """Return initial predictions in the same order as ``indices``."""

    predictions: list[int] = []
    with torch.inference_mode():
        for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
            predictions.extend(model(images).argmax(dim=1).cpu().tolist())
    return [int(value) for value in predictions]


def run_grid(
    model,
    dataset,
    indices: list[int],
    *,
    model_name: str,
    target: int,
    steps: int,
    restarts: int,
    random_start: bool,
    batch_size: int,
    device: torch.device,
    phase: str,
) -> tuple[list[dict], dict[int, float], dict[int, bool]]:
    """Run one PGD configuration over an epsilon grid and return long records."""

    before = initial_predictions(model, dataset, indices, batch_size=batch_size, device=device)
    first_success: dict[int, float] = {
        index: 0.0 for index, prediction in zip(indices, before) if prediction == int(target)
    }
    found: dict[int, bool] = {index: prediction == int(target) for index, prediction in zip(indices, before)}
    records: list[dict] = []

    for epsilon in (0.0,) + EPSILON_GRID:
        if epsilon == 0.0:
            successes = torch.tensor([prediction == int(target) for prediction in before], dtype=torch.bool)
            actual_norm = torch.zeros(len(indices), dtype=torch.float32)
            after = torch.tensor(before, dtype=torch.long)
        else:
            successes_list: list[torch.Tensor] = []
            norms_list: list[torch.Tensor] = []
            predictions_list: list[torch.Tensor] = []
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
                successes_list.append(result.success.detach().cpu())
                norms_list.append(result.best_linf.detach().cpu())
                predictions_list.append(result.best_prediction.detach().cpu())
            successes = torch.cat(successes_list)
            actual_norm = torch.cat(norms_list)
            after = torch.cat(predictions_list)

        for offset, sample_index in enumerate(indices):
            success = bool(successes[offset])
            if success and not found[sample_index]:
                first_success[sample_index] = float(actual_norm[offset])
                found[sample_index] = True
            records.append(
                {
                    "phase": phase,
                    "model_name": model_name,
                    "sample_index": int(sample_index),
                    "epsilon": float(epsilon),
                    "epsilon_pixels": float(epsilon * 255.0),
                    "restart": int(restarts),
                    "before_prediction": int(before[offset]),
                    "after_prediction": int(after[offset]),
                    "success": success,
                    "actual_best_linf": float(actual_norm[offset]),
                    "target": int(target),
                }
            )
    return records, first_success, found


def sorted_robust_indices(indices: list[int], radius: dict[int, float], before: list[int], target: int, count: int) -> list[int]:
    """Select largest-radius samples, placing right-censored samples first."""

    eligible = [index for index, prediction in zip(indices, before) if prediction != int(target)]
    eligible.sort(key=lambda index: (-float(radius.get(index, float("inf"))), int(index)))
    return eligible[: int(count)]


def first_success_by_sample(records: list[dict]) -> dict[int, float]:
    """Recover the first finite successful epsilon from long attack records."""

    output: dict[int, float] = {}
    for row in records:
        if bool(row["success"]) and int(row["sample_index"]) not in output:
            output[int(row["sample_index"])] = float(row["epsilon"])
    return output


def make_plots(output: Path, evaluation_rows: list[dict]) -> None:
    """Create compact PNG figures; failure to import plotting is non-fatal."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in evaluation_rows:
        key = (row["selection_clean_seed"], row["model_group"])
        grouped.setdefault(key, []).append(row)
    for (seed, group), rows in grouped.items():
        values = sorted({float(row["epsilon_pixels"]) for row in rows})
        asr = [np.mean([row["success"] for row in rows if float(row["epsilon_pixels"]) == value]) for value in values]
        plt.plot(values, asr, marker="o", label=f"{group}-seed{seed}")
    plt.xlabel("epsilon (pixel units / 255)")
    plt.ylabel("targeted ASR to class 0")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "figures" / "asr_curves.png", dpi=160)
    plt.close()


def main() -> None:
    """Execute Stage 1A and write the complete pilot result directory."""

    args = parse_args()
    seed_everything(args.candidate_seed)
    device = torch.device(args.device)
    model_root = Path(args.model_root)
    clean_seeds = ints(args.clean_seeds)
    backdoor_groups = [item.strip() for item in args.backdoor_groups.split(",") if item.strip()]
    output = timestamp_run_dir(args.output_root, "stage1a")

    candidate_dataset = cifar100_dataset(args.data_root, train=False)
    order = np.random.default_rng(args.candidate_seed).permutation(len(candidate_dataset)).tolist()
    candidate_indices = [int(index) for index in order[: int(args.candidate_count)]]
    write_csv(output / "candidate_pool.csv", [{"pool_position": n, "sample_index": index, "dataset": "CIFAR100_test"} for n, index in enumerate(candidate_indices)])

    specs: list[dict] = []
    for seed in clean_seeds:
        specs.append({"group": "clean", "kind": "clean", "seed": seed, "path": checkpoint_path(model_root, args.clean_group, seed)})
        for group in backdoor_groups:
            specs.append({"group": group, "kind": "backdoor", "seed": seed, "path": checkpoint_path(model_root, group, seed)})
    coarse_rows: list[dict] = []
    refined_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    selection_summary: list[dict] = []
    selected_by_seed: dict[int, list[int]] = {}
    clean_models: dict[int, object] = {}

    for seed in clean_seeds:
        clean_spec = next(spec for spec in specs if spec["kind"] == "clean" and spec["seed"] == seed)
        clean_model, _ = load_model(clean_spec["path"], args.backdoorbench_root, device)
        clean_models[seed] = clean_model
        before = initial_predictions(clean_model, candidate_dataset, candidate_indices, batch_size=args.batch_size, device=device)
        rows, coarse_radius, _ = run_grid(
            clean_model, candidate_dataset, candidate_indices, model_name=f"clean_seed{seed}", target=args.target,
            steps=30, restarts=1, random_start=False, batch_size=args.batch_size, device=device, phase="coarse",
        )
        coarse_rows.extend([{**row, "selection_clean_seed": seed} for row in rows])
        coarse_top = sorted_robust_indices(candidate_indices, coarse_radius, before, args.target, args.top_coarse)
        refined, refined_radius, _ = run_grid(
            clean_model, candidate_dataset, coarse_top, model_name=f"clean_seed{seed}", target=args.target,
            steps=100, restarts=3, random_start=True, batch_size=args.batch_size, device=device, phase="refine",
        )
        refined_rows.extend([{**row, "selection_clean_seed": seed, "coarse_top40": True} for row in refined])
        selected = sorted_robust_indices(coarse_top, refined_radius, [before[candidate_indices.index(index)] for index in coarse_top], args.target, args.top_final)
        selected_by_seed[seed] = selected
        for rank, index in enumerate(selected, start=1):
            selection_summary.append(
                {
                    "selection_clean_seed": seed,
                    "sample_index": index,
                    "final_rank": rank,
                    "refined_radius": refined_radius.get(index, float("inf")),
                    "censored_gt_32_255": not np.isfinite(refined_radius.get(index, float("inf"))),
                }
            )

        evaluation_specs = [clean_spec] + [spec for spec in specs if spec["kind"] == "backdoor" and spec["seed"] == seed]
        for spec in evaluation_specs:
            model = clean_model if spec["kind"] == "clean" else load_model(spec["path"], args.backdoorbench_root, device)[0]
            evaluated, _, _ = run_grid(
                model, candidate_dataset, selected, model_name=f"{spec['group']}_seed{seed}", target=args.target,
                steps=100, restarts=3, random_start=True, batch_size=args.batch_size, device=device, phase="paired_evaluation",
            )
            evaluation_rows.extend([{**row, "selection_clean_seed": seed, "model_group": spec["group"]} for row in evaluated])

    write_csv(output / "coarse_attack.csv", coarse_rows)
    write_csv(output / "refined_attack.csv", refined_rows)
    write_csv(output / "selection_summary.csv", selection_summary)
    write_csv(output / "paired_evaluation.csv", evaluation_rows)

    comparisons = []
    for seed in clean_seeds:
        clean_rows = [row for row in evaluation_rows if row["selection_clean_seed"] == seed and row["model_group"] == "clean"]
        clean_radius = first_success_by_sample(clean_rows)
        for group in backdoor_groups:
            bd_rows = [row for row in evaluation_rows if row["selection_clean_seed"] == seed and row["model_group"] == group]
            bd_radius = first_success_by_sample(bd_rows)
            delta_rows = [clean_radius[index] - bd_radius[index] for index in selected_by_seed[seed] if index in clean_radius and index in bd_radius]
            positive_adjacent = 0
            for left, right in zip(EPSILON_GRID[:-1], EPSILON_GRID[1:]):
                clean_asr_left = np.mean([row["success"] for row in clean_rows if row["epsilon"] == left])
                bd_asr_left = np.mean([row["success"] for row in bd_rows if row["epsilon"] == left])
                clean_asr_right = np.mean([row["success"] for row in clean_rows if row["epsilon"] == right])
                bd_asr_right = np.mean([row["success"] for row in bd_rows if row["epsilon"] == right])
                positive_adjacent += int(bd_asr_left > clean_asr_left and bd_asr_right > clean_asr_right)
            comparisons.append(
                {
                    "selection_clean_seed": seed,
                    "backdoor_group": group,
                    "positive_adjacent_intervals": positive_adjacent,
                    "median_delta_r_found_pairs": float(np.median(delta_rows)) if delta_rows else None,
                    "found_pair_count": len(delta_rows),
                }
            )
    summary = {
        "protocol": "stage1a-oracle-bridge-v1",
        "target": args.target,
        "candidate_count": len(candidate_indices),
        "epsilon_pixels": [1, 2, 4, 8, 16, 32],
        "coarse": {"steps": 30, "restarts": 1, "random_start": False},
        "refine_and_evaluation": {"steps": 100, "restarts": 3, "random_start": True},
        "clean_seeds": clean_seeds,
        "backdoor_groups": backdoor_groups,
        "selections": selected_by_seed,
        "comparisons": comparisons,
    }
    write_json(output / "summary.json", summary)
    make_plots(output, evaluation_rows)
    print(f"Stage 1A complete: {output.resolve()}")


if __name__ == "__main__":
    main()
