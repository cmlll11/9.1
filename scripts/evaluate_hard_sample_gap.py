"""Select Clean-hard CIFAR-10 probes and evaluate all trained GAP mappings."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from mdluap.data import cifar10_dataset
from mdluap.gap import build_official_gap_generator
from mdluap.mappings import ImageDependentResidualMapping
from mdluap.models import load_attack_result_model


def parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--selection-group", default="clean_select")
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--gap-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--mapping-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--gate-report", required=False)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--epsilon", type=parse_fraction, default=parse_fraction("4/255"))
    parser.add_argument("--selection-seeds", default="0,1,2,3,4")
    parser.add_argument("--evaluation-seeds", default="5,6,7,8,9")
    parser.add_argument("--backdoor-seeds", default="0,1,2,3,4")
    parser.add_argument("--hard-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def model_specs(args: argparse.Namespace) -> list[dict]:
    root = Path(args.model_root)
    specs = []
    for seed in ints(args.selection_seeds):
        specs.append({"group": "clean_select", "artifact_group": args.selection_group, "trigger": "clean", "seed": seed, "partition": "shared", "result": root / args.selection_group / f"seed{seed}" / "attack_result.pt"})
    for seed in ints(args.evaluation_seeds):
        specs.append({"group": "clean_eval", "trigger": "clean", "seed": seed, "partition": "shared", "result": root / "clean_eval" / f"seed{seed}" / "attack_result.pt"})
    for trigger in ("badnet", "lf", "blended", "wanet"):
        for seed in ints(args.backdoor_seeds):
            specs.append({"group": trigger, "trigger": trigger, "seed": seed, "partition": "shared", "result": root / trigger / f"seed{seed}" / "attack_result.pt"})
    return specs


def mapping_path(root: Path, spec: dict) -> Path:
    return root / spec.get("artifact_group", spec["group"]) / f"seed{spec['seed']}" / "mapping.pt"


def load_mapping(path: Path, args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator = build_official_gap_generator(
        gap_root=args.gap_root,
        device=device,
        ngf=int(checkpoint.get("ngf", 64)),
        output_channels=3,
    )
    mapping = ImageDependentResidualMapping(generator, float(checkpoint["epsilon"])).to(device)
    mapping.load_state_dict(checkpoint["mapping"], strict=True)
    return mapping.eval()


def target_margin(logits: torch.Tensor, target: int) -> torch.Tensor:
    other = logits.clone()
    other[:, int(target)] = float("-inf")
    return logits[:, int(target)] - other.max(dim=1).values


@torch.inference_mode()
def score_model(
    spec: dict,
    mapping_path_value: Path,
    samples,
    args: argparse.Namespace,
    device: torch.device,
    gate_status: str = "unknown",
) -> tuple[dict, list[dict]]:
    model_wrapper, _ = load_attack_result_model(
        str(spec["result"]), backdoorbench_root=args.backdoorbench_root, device=device
    )
    mapping = load_mapping(mapping_path_value, args, device)
    data_loader = DataLoader(samples, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    target = int(args.target)
    sample_rows: list[dict] = []
    eligible_count = baseline_target_count = success_count = 0
    before_probs: list[float] = []
    after_probs: list[float] = []
    before_margins: list[float] = []
    after_margins: list[float] = []

    for offset, (images, labels) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        before_logits = model_wrapper(images)
        after_logits = model_wrapper(mapping(images))
        before_pred = before_logits.argmax(dim=1)
        after_pred = after_logits.argmax(dim=1)
        eligible = (labels != target) & (before_pred != target)
        success = eligible & (after_pred == target)
        baseline_target_count += int(((labels != target) & (before_pred == target)).sum())
        eligible_count += int(eligible.sum())
        success_count += int(success.sum())

        before_prob = F.softmax(before_logits, dim=1)[:, target]
        after_prob = F.softmax(after_logits, dim=1)[:, target]
        before_margin = target_margin(before_logits, target)
        after_margin = target_margin(after_logits, target)
        for local_index in range(images.shape[0]):
            if bool(eligible[local_index]):
                before_probs.append(float(before_prob[local_index]))
                after_probs.append(float(after_prob[local_index]))
                before_margins.append(float(before_margin[local_index]))
                after_margins.append(float(after_margin[local_index]))
            sample_rows.append(
                {
                    "sample_index": offset * args.batch_size + local_index,
                    "eligible": bool(eligible[local_index]),
                    "baseline_target": bool(before_pred[local_index] == target),
                    "success": bool(success[local_index]),
                    "target_probability_before": float(before_prob[local_index]),
                    "target_probability_after": float(after_prob[local_index]),
                    "target_margin_before": float(before_margin[local_index]),
                    "target_margin_after": float(after_margin[local_index]),
                }
            )

    def average(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    result = {
        "model_group": spec["group"],
        "trigger": spec["trigger"],
        "seed": spec["seed"],
        "partition": spec["partition"],
        "n_samples": len(samples),
        "eligible_count": eligible_count,
        "baseline_target_count": baseline_target_count,
        "baseline_target_rate": baseline_target_count / max(len(samples), 1),
        "success_count": success_count,
        "success_rate": success_count / max(eligible_count, 1),
        "target_probability_before_mean": average(before_probs),
        "target_probability_after_mean": average(after_probs),
        "target_probability_increase_mean": average([b - a for a, b in zip(before_probs, after_probs)]),
        "target_margin_before_mean": average(before_margins),
        "target_margin_after_mean": average(after_margins),
        "target_margin_increase_mean": average([b - a for a, b in zip(before_margins, after_margins)]),
        "status": "complete",
        "gate_status": gate_status,
    }
    del model_wrapper, mapping
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, sample_rows


def select_hard_samples(clean_rows: list[dict], samples, args: argparse.Namespace) -> list[int]:
    candidates = []
    selection_models = len({(row["model_group"], row["seed"]) for row in clean_rows})
    for index in range(len(samples)):
        rows = [row for row in clean_rows if row["sample_index"] == index]
        if len(rows) != selection_models:
            continue
        if int(samples[index][1]) == args.target:
            continue
        if any(row["baseline_target"] or not row["eligible"] for row in rows):
            continue
        if any(row["success"] for row in rows):
            continue
        mean_probability = sum(row["target_probability_after"] for row in rows) / len(rows)
        mean_margin = sum(row["target_margin_after"] for row in rows) / len(rows)
        candidates.append((mean_probability, mean_margin, int(samples[index][1]), index))
    candidates.sort()
    selected: list[int] = []
    per_class = {label: 0 for label in range(10) if label != args.target}
    quota = args.hard_count // len(per_class)
    for _, _, label, index in candidates:
        if len(selected) >= args.hard_count:
            break
        if per_class[label] < quota:
            selected.append(index)
            per_class[label] += 1
    for _, _, _, index in candidates:
        if len(selected) >= args.hard_count:
            break
        if index not in selected:
            selected.append(index)
    if len(selected) < args.hard_count:
        raise RuntimeError(f"hard_pool_insufficient: found {len(selected)}, need {args.hard_count}")
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    specs = model_specs(args)
    model_root = Path(args.model_root)
    mapping_root = Path(args.mapping_root)
    gate_rows = {}
    if args.gate_report:
        gate_payload = json.loads(Path(args.gate_report).read_text(encoding="utf-8"))
        gate_rows = gate_payload.get("models", {})
    for spec in specs:
        if not Path(spec["result"]).is_file():
            raise FileNotFoundError(spec["result"])
        if not mapping_path(mapping_root, spec).is_file():
            raise FileNotFoundError(mapping_path(mapping_root, spec))

    candidate_pool = cifar10_dataset(args.candidate_root, train=True)
    selection_specs = [spec for spec in specs if spec["group"] == "clean_select"]
    failed_selection = [
        f"{spec['group']}_seed{spec['seed']}"
        for spec in selection_specs
        if gate_rows.get(f"{spec['group']}_seed{spec['seed']}", {}).get("status") == "gate_failed"
    ]
    if failed_selection:
        # Selection models are only used to define hard samples.  Their
        # accuracy is still recorded, but they do not need to pass the
        # strict final-model gate used for clean_eval and backdoor models.
        print(f"Selection Clean models retained for hard-sample screening: {failed_selection}")
    clean_scores = []
    for spec in selection_specs:
        gate_status = gate_rows.get(f"{spec['group']}_seed{spec['seed']}", {}).get("status", "unknown")
        _, rows = score_model(spec, mapping_path(mapping_root, spec), candidate_pool, args, device, gate_status)
        clean_scores.extend([{**row, "model_group": spec["group"], "seed": spec["seed"]} for row in rows])
    selected_indices = select_hard_samples(clean_scores, candidate_pool, args)
    hard_set = Subset(candidate_pool, selected_indices)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    hard_metadata = []
    for index in selected_indices:
        per_model = [row for row in clean_scores if row["sample_index"] == index]
        hard_metadata.append(
            {
                "sample_index": index,
                "true_label": int(candidate_pool[index][1]),
                "clean_selection_success_count": sum(int(row["success"]) for row in per_model),
                "clean_selection_success_rate": 0.0,
                "mean_target_probability": sum(row["target_probability_after"] for row in per_model) / len(per_model),
                "mean_target_margin": sum(row["target_margin_after"] for row in per_model) / len(per_model),
            }
        )
    for item in hard_metadata:
        item["clean_selection_success_rate"] = item["clean_selection_success_count"] / len(selection_specs)
    (output_root / "hard_samples.json").write_text(
        json.dumps({"protocol": "hard-sample-gap-v1", "indices": selected_indices, "samples": hard_metadata}, indent=2),
        encoding="utf-8",
    )
    (output_root / "clean_selection_scores.json").write_text(json.dumps(clean_scores), encoding="utf-8")

    results = []
    per_sample = {}
    for spec in specs:
        gate_status = gate_rows.get(f"{spec['group']}_seed{spec['seed']}", {}).get("status", "unknown")
        result, sample_rows = score_model(spec, mapping_path(mapping_root, spec), hard_set, args, device, gate_status)
        results.append(result)
        per_sample[f"{spec['group']}_seed{spec['seed']}"] = sample_rows
    grouped = {}
    for row in results:
        group = "clean" if row["model_group"] == "clean_eval" else row["model_group"]
        if row.get("gate_status") not in {"gate_failed"}:
            grouped.setdefault(group, []).append(row)
    aggregates = {}
    for group, group_rows in grouped.items():
        aggregates[group] = {"n_models": len(group_rows)}
        for key in ("success_count", "success_rate", "target_probability_increase_mean", "target_margin_increase_mean"):
            values = [row[key] for row in group_rows if row[key] is not None]
            aggregates[group][f"{key}_mean"] = statistics.mean(values) if values else None
            aggregates[group][f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    summary = {
        "protocol": "hard-sample-gap-v1",
        "dataset": "CIFAR-10",
        "hard_pool": "heldout_cifar10_train_partition",
        "candidate_pool_size": len(candidate_pool),
        "epsilon_pixels": 4,
        "target": args.target,
        "hard_count": len(selected_indices),
        "results": results,
        "aggregates": aggregates,
        "per_sample": per_sample,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(Path(args.csv), results)
    print(f"Hard samples complete: count={len(selected_indices)}")
    print(f"Summary complete: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
