"""Stage 1D-WT: wrong-target PGD and official trigger-direction analysis.

Clean seeds 1--3 train one Ridge Probe for each wrong target.  Clean seed 0
uses each Probe to select one shared CIFAR-100 Top-100 set.  That same set is
then sent to Clean0 and each seed-0 backdoor model so the mechanism comparison
is paired by image.  The attack target is 1, 3, or 7, while the true backdoor
target remains 0.

This script deliberately does not claim a literal class-transition path.  It
tests whether a successful wrong-target PGD endpoint has a feature change
direction similar to the official trigger transformation for that attack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mdluap.data import cifar100_dataset
from mdluap.models import load_modelzoo_classifier
from mdluap.official_triggers import build_trigger_adapters
from mdluap.probes import RidgeProbe, target_conditioned_logits_features, target_feature_names, target_margin
from mdluap.targeted_pgd import targeted_pgd, targeted_pgd_endpoint
from pilot_common import batch_images, seed_everything, timestamp_run_dir, write_csv, write_json


TRAIN_EPS_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
ANALYSIS_EPS_PIXELS = (1.0, 1.5)
DEFAULT_TARGETS = (1, 3, 7)
BACKDOOR_GROUPS = ("badnet", "blended", "wanet", "inputaware", "ssba", "adaptive_blend")
SHARED_TRIGGER_GROUPS = {"badnet", "blended", "wanet", "adaptive_blend"}
SAMPLE_TRIGGER_GROUPS = {"ssba", "inputaware"}


class AvgPoolFeatures:
    """Capture the 512-dimensional classifier avgpool output."""

    def __init__(self, classifier: nn.Module):
        backbone = getattr(classifier, "model", classifier)
        layer = getattr(backbone, "avgpool", None)
        self._use_spatial_fallback = layer is None
        if self._use_spatial_fallback:
            layer = getattr(backbone, "layer4", None)
        if layer is None:
            raise AttributeError("model must expose avgpool or layer4 for Stage 1D-WT")
        self.value: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._hook)
        self.classifier = classifier

    def _hook(self, _module, _inputs, output):
        self.value = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(images).detach()
        if self.value is None:
            raise RuntimeError("avgpool hook did not capture a feature")
        value = self.value
        if self._use_spatial_fallback and value.ndim == 4:
            value = torch.nn.functional.adaptive_avg_pool2d(value, 1)
        return logits, value.flatten(1).detach()

    def close(self) -> None:
        self.handle.remove()


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or tuple(sorted(values)) != values or any(value <= 0 for value in values):
        raise ValueError("epsilon values must be a positive ascending list")
    return values


def model_alias(group: str, seed: int) -> str:
    if group == "clean":
        return f"clean{seed}"
    aliases = {
        "badnet": "badnet0",
        "blended": "blended0",
        "wanet": "wanet0",
        "inputaware": "inputaware0",
        "ssba": "ssba0",
        "adaptive_blend": "adaptive_blend01",
    }
    try:
        return aliases[group]
    except KeyError as exc:
        raise ValueError(f"no Model Zoo alias configured for group {group!r}") from exc


def load_any_model(model_zoo_root: Path, group: str, seed: int, *, device: torch.device):
    alias = model_alias(group, seed)
    return load_modelzoo_classifier(alias, model_zoo_root=str(model_zoo_root), device=device)


@torch.inference_mode()
def logits_and_features(model, dataset, indices: list[int], *, target: int, batch_size: int, device: torch.device):
    logits_rows, feature_rows, labels = [], [], []
    for _, images, batch_labels in batch_images(dataset, indices, batch_size=batch_size, device=device):
        logits = model(images).detach()
        logits_rows.append(logits.cpu())
        feature_rows.append(target_conditioned_logits_features(logits, target).cpu().numpy())
        labels.extend(batch_labels.cpu().tolist())
    logits = torch.cat(logits_rows)
    return logits, np.concatenate(feature_rows), np.asarray(labels, dtype=np.int64)


def coarse_probe_labels(model, dataset, indices: list[int], *, target: int, eps_pixels: tuple[float, ...], steps: int, batch_size: int, device: torch.device, seed: int):
    """Create full-grid first-success labels for one reference Clean model."""

    logits, _, _ = logits_and_features(model, dataset, indices, target=target, batch_size=batch_size, device=device)
    before = logits.argmax(dim=1).numpy()
    eligible = before != int(target)
    # Samples already predicted as the wrong target are not attack trials.
    # They remain in the pool for provenance, but never become epsilon=0
    # labels and never enter the Ridge fit.
    first = np.full(len(indices), np.inf, dtype=np.float64)
    rows = []
    for eps in eps_pixels:
        successes = np.zeros(len(indices), dtype=bool)
        predictions = np.full(len(indices), -1, dtype=np.int64)
        offset = 0
        for _, images, _ in batch_images(dataset, indices, batch_size=batch_size, device=device):
            local_count = len(images)
            local_eligible = eligible[offset : offset + local_count]
            if np.any(local_eligible):
                mask = torch.as_tensor(local_eligible, device=images.device)
                result = targeted_pgd(model, images[mask], target=target, epsilon=eps / 255.0, steps=steps,
                                      alpha=eps / 255.0 / 10.0, random_start=False, restarts=1)
                positions = np.flatnonzero(local_eligible)
                successes[offset + positions] = result.success.cpu().numpy()
                predictions[offset + positions] = result.best_prediction.cpu().numpy()
            offset += local_count
        for pos, sample_index in enumerate(indices):
            if not eligible[pos]:
                rows.append({
                    "phase": "probe_label", "reference_clean_seed": seed, "target": target,
                    "sample_index": sample_index, "epsilon_pixels": eps,
                    "before_prediction": int(before[pos]), "after_prediction": None,
                    "eligible": False, "pgd_status": "ineligible_original_target", "success": None,
                    "robustness_radius_pixels": None, "censored": None, "steps": steps,
                    "random_start": False, "restarts": 1,
                })
                continue
            success = bool(successes[pos])
            if success and not math.isfinite(first[pos]):
                first[pos] = eps
            rows.append({
                "phase": "probe_label", "reference_clean_seed": seed, "target": target,
                "sample_index": sample_index, "epsilon_pixels": eps,
                "before_prediction": int(before[pos]), "after_prediction": int(predictions[pos]),
                "eligible": True, "pgd_status": "success" if success else "failure", "success": success,
                "robustness_radius_pixels": None if not math.isfinite(first[pos]) else float(first[pos]),
                "censored": not math.isfinite(first[pos]), "steps": steps, "random_start": False, "restarts": 1,
            })
    return first, rows


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(len(left), np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = (left[valid] * right[valid]).sum(axis=1) / denominator[valid]
    return result


def cosine_to_vector(vectors: np.ndarray, vector: np.ndarray | None) -> np.ndarray:
    if vector is None:
        return np.full(len(vectors), np.nan, dtype=np.float64)
    return cosine_rows(vectors, np.broadcast_to(vector, vectors.shape))


def concentration(vectors: np.ndarray) -> float | None:
    if len(vectors) < 2:
        return None
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    values = (normalized @ normalized.T)[np.triu_indices(len(vectors), k=1)]
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None


def prototype(vectors: np.ndarray) -> np.ndarray | None:
    """Return the normalized mean direction for a shared-trigger summary."""

    if len(vectors) == 0:
        return None
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    value = normalized.mean(axis=0)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-12 else None


def shuffled_alignment(adv: np.ndarray, trigger: np.ndarray, *, seed: int, repeats: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(repeats)):
        order = rng.permutation(len(trigger))
        while len(trigger) > 1 and np.any(order == np.arange(len(trigger))):
            order = rng.permutation(len(trigger))
        values.append(cosine_rows(adv, trigger[order]))
    return np.nanmean(np.stack(values), axis=0)


def backdoor_control_cohort(eligible: np.ndarray, trigger_success: np.ndarray) -> np.ndarray:
    """Select the paired cohort once from the Backdoor model only."""

    return np.asarray(eligible, dtype=bool) & np.asarray(trigger_success, dtype=bool)


def alignment_definition(trigger_type: str) -> str:
    """Return the protocol-specific trigger alignment definition."""

    if trigger_type in SHARED_TRIGGER_GROUPS:
        return "prototype"
    if trigger_type in SAMPLE_TRIGGER_GROUPS:
        return "same_pair"
    raise ValueError(f"unknown trigger family: {trigger_type}")


def probe_topk_positions(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return the shared Probe Top-k positions without model-specific filtering.

    Eligibility is deliberately evaluated later for every model.  Filtering
    here would silently reselect the shared cohort whenever Clean0 already
    predicts a wrong target, which would violate the fixed Top-100 protocol.
    """

    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("Probe scores must be a one-dimensional array")
    return np.argsort(-scores, kind="mergesort")[: int(top_k)]


def write_figures(output: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    usable = [row for row in rows if row.get("alignment_available") and row.get("same_alignment") is not None]
    if usable:
        groups = defaultdict(list)
        for row in usable:
            groups[f"{row['model_alias']}:{row['trigger_type']}:{row['sample_group']}"] .append(float(row["same_alignment"]))
        labels = list(groups)
        plt.figure(figsize=(max(10, len(labels) * 0.6), 5))
        plt.boxplot([groups[label] for label in labels], labels=labels, showfliers=False)
        plt.xticks(rotation=65, ha="right")
        plt.ylabel("trigger-related avgpool alignment")
        plt.tight_layout()
        plt.savefig(output / "figures/alignment_boxplot.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-zoo-root", default=None)
    parser.add_argument("--model-zoo-source-root", default=None)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--trigger-artifact-root", default=None)
    parser.add_argument("--output-root", default="results/stage1d_wrong_target_trigger_alignment")
    parser.add_argument("--clean-group", default="clean_select_shared")
    parser.add_argument("--backdoor-groups", default=",".join(BACKDOOR_GROUPS))
    parser.add_argument("--reference-clean-seeds", default="1,2,3")
    parser.add_argument("--test-clean-seed", type=int, default=0)
    parser.add_argument("--targets", default="1,3,7")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=2031)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-eps-pixels", default=",".join(map(str, TRAIN_EPS_PIXELS)))
    parser.add_argument("--analysis-eps-pixels", default=",".join(map(str, ANALYSIS_EPS_PIXELS)))
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--analysis-steps", type=int, default=100)
    parser.add_argument("--analysis-restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ssba-encoder-path", default=None)
    parser.add_argument("--ssba-config-path", default=None)
    parser.add_argument("--inputaware-state-path", default=None)
    parser.add_argument("--adaptive-blend-trigger-path", default=None)
    parser.add_argument("--blended-alpha", type=float, default=0.2)
    parser.add_argument("--wanet-s", type=float, default=0.5)
    parser.add_argument("--wanet-grid-rescale", type=float, default=1.0)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--attack-seed", type=int, default=2031)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


MODEL_ZOO_ALIASES = (
    "clean0", "clean1", "clean2", "clean3", "badnet0", "blended0",
    "wanet0", "inputaware0", "ssba0", "adaptive_blend01",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def model_zoo_preflight(root: Path, aliases: tuple[str, ...], device: torch.device) -> dict:
    """Validate aliases and capture complete Model Zoo provenance.

    The preflight deliberately uses only the Model Zoo public API for model
    loading.  Paths returned by ``get_model_info`` are checked for the
    expected artifacts, but checkpoints are never loaded directly here.
    """

    try:
        import modelzoo
        from modelzoo import get_model_info, list_models
    except ImportError as exc:  # pragma: no cover - server-only dependency
        raise ImportError("backdoor-model-zoo must be installed before Stage 1D-WT") from exc

    try:
        registered = list_models(root=str(root))
    except TypeError:
        # Version 0.1.0 exposes list_models() without a root parameter and
        # reads MODEL_ZOO_ROOT from the environment.
        os.environ["MODEL_ZOO_ROOT"] = str(root)
        registered = list_models()
    registered_aliases = {
        str(row.get("alias")) for row in registered
        if isinstance(row, dict) and row.get("alias") is not None
    }
    missing = sorted(set(aliases) - registered_aliases)
    if missing:
        raise ValueError(f"Model Zoo aliases are missing: {missing}")

    os.environ["MODEL_ZOO_ROOT"] = str(root)
    infos: dict[str, dict] = {}
    for alias in aliases:
        try:
            info = get_model_info(alias)
        except TypeError:
            info = get_model_info(alias, root=str(root))
        if not isinstance(info, dict):
            raise TypeError(f"get_model_info({alias!r}) did not return a dictionary")
        model_directory = Path(str(info.get("model_directory", "")))
        artifact_paths = {
            "checkpoint": info.get("checkpoint"),
            "config": info.get("config_path") or (model_directory / "config.yaml"),
            "metadata": info.get("metadata_path") or (model_directory / "metadata.json"),
        }
        for key, value in artifact_paths.items():
            if value is None or not Path(value).is_file():
                raise FileNotFoundError(f"Model Zoo {alias} {key} is missing: {value}")
        enriched = dict(info)
        enriched["_artifact_paths"] = {key: str(Path(value).resolve()) for key, value in artifact_paths.items()}
        infos[alias] = enriched

    # Loading once in preflight verifies the public loader, model architecture,
    # eval mode, and output dimensionality before the expensive experiment.
    for alias in aliases:
        model = load_modelzoo_classifier(alias, model_zoo_root=str(root), device=device)[0]
        with torch.no_grad():
            logits = model(torch.zeros(2, 3, 32, 32, device=device))
        if tuple(logits.shape) != (2, 10):
            raise ValueError(f"Model Zoo model {alias} returned {tuple(logits.shape)}, expected [2,10]")

    registry_candidates = [root / "registry.yaml", root / "registry.yml"]
    registry_path = next((path for path in registry_candidates if path.is_file()), None)
    if registry_path is None:
        matches = sorted(root.rglob("registry.yaml"))
        registry_path = matches[0] if matches else None
    package_version = getattr(modelzoo, "__version__", None)
    if package_version is None:
        try:
            package_version = importlib.metadata.version("backdoor-model-zoo")
        except importlib.metadata.PackageNotFoundError:
            package_version = None
    return {
        "package_version": package_version,
        "registered_aliases": sorted(registered_aliases),
        "list_models": registered,
        "registry_path": str(registry_path.resolve()) if registry_path else None,
        "registry_sha256": _sha256(registry_path) if registry_path else None,
        "infos": infos,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.attack_seed)
    targets = parse_ints(args.targets)
    if not targets or any(target == 0 or target < 0 or target > 9 for target in targets):
        raise ValueError("Stage 1D-WT targets must be nonzero CIFAR-10 classes")
    reference_seeds = parse_ints(args.reference_clean_seeds)
    if args.test_clean_seed in reference_seeds:
        raise ValueError("test Clean seed must be excluded from Probe reference seeds")
    train_eps = parse_floats(args.train_eps_pixels)
    analysis_eps = parse_floats(args.analysis_eps_pixels)
    if tuple(analysis_eps) != ANALYSIS_EPS_PIXELS:
        raise ValueError("Stage 1D-WT mechanism analysis is fixed to epsilon 1 and 1.5 / 255")
    groups = tuple(item.strip() for item in args.backdoor_groups.split(",") if item.strip())
    if set(groups) - set(BACKDOOR_GROUPS):
        raise ValueError(f"unsupported backdoor groups: {sorted(set(groups) - set(BACKDOOR_GROUPS))}")

    raw_model_zoo_root = args.model_zoo_root or os.environ.get("MODEL_ZOO_ROOT")
    if not raw_model_zoo_root:
        raise ValueError("--model-zoo-root or MODEL_ZOO_ROOT is required")
    model_zoo_root = Path(raw_model_zoo_root).expanduser()
    if not model_zoo_root.is_dir():
        raise FileNotFoundError(f"Model Zoo root does not exist: {model_zoo_root}")
    output = timestamp_run_dir(args.output_root, "wrong_target_alignment")
    device = torch.device(args.device)
    data_root, bdb_root = Path(args.data_root), Path(args.backdoorbench_root)
    artifact_root = Path(args.trigger_artifact_root) if args.trigger_artifact_root else bdb_root
    source_root_value = args.model_zoo_source_root or os.environ.get("MODEL_ZOO_SOURCE_ROOT")
    source_root = Path(source_root_value).expanduser() if source_root_value else None
    all_aliases = MODEL_ZOO_ALIASES
    model_zoo_provenance = model_zoo_preflight(model_zoo_root, all_aliases, device)
    train_data, test_data = cifar100_dataset(data_root, train=True), cifar100_dataset(data_root, train=False)
    rng = np.random.default_rng(args.split_seed)
    train_indices = rng.permutation(len(train_data))[:args.train_count].astype(int).tolist()
    test_indices = rng.permutation(len(test_data))[:args.test_count].astype(int).tolist()
    write_csv(output / "candidate_pool.csv", [
        {"split": "cifar100_train", "position": pos, "sample_index": index} for pos, index in enumerate(train_indices)
    ] + [{"split": "cifar100_test", "position": pos, "sample_index": index} for pos, index in enumerate(test_indices)])
    config = vars(args).copy()
    config.update({
        "protocol": "stage1d-wt-modelzoo-control-cohort-v2",
        "targets": list(targets),
        "reference_clean_seeds": list(reference_seeds),
        "analysis_eps_pixels": list(analysis_eps),
        "train_eps_pixels": list(train_eps),
        "model_aliases": {"clean0": "clean0", "clean_reference": [f"clean{seed}" for seed in reference_seeds], "backdoor": {group: model_alias(group, 0) for group in groups}},
        "model_zoo_root": str(model_zoo_root.resolve()),
        "model_zoo_source_root": str(source_root.resolve()) if source_root else None,
        "model_zoo_git_commit": _git_commit(source_root),
        "model_zoo_provenance": model_zoo_provenance,
        "control_cohort": "backdoor_eligible_and_backdoor_trigger_success",
    })
    output.joinpath("config.resolved.yaml").write_text(__import__("yaml").safe_dump(config, sort_keys=False), encoding="utf-8")
    log_path = output / "run.log"
    def log(message: str):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    log(f"Stage 1D-WT started: {output.resolve()}")

    reference_models = {seed: load_any_model(model_zoo_root, "clean", seed, device=device)[0] for seed in reference_seeds}
    probes, probe_rows = {}, []
    for target in targets:
        feature_rows, label_rows = [], []
        names = target_feature_names(target, 10)
        for seed, model in reference_models.items():
            logits, features, _ = logits_and_features(model, train_data, train_indices, target=target, batch_size=args.batch_size, device=device)
            radius, coarse_rows = coarse_probe_labels(model, train_data, train_indices, target=target, eps_pixels=train_eps, steps=args.train_steps, batch_size=args.batch_size, device=device, seed=seed)
            finite = np.isfinite(radius) & (logits.argmax(dim=1).numpy() != target)
            feature_rows.extend(features[finite])
            label_rows.extend(radius[finite])
            for pos, sample_index in enumerate(train_indices):
                eligible = bool(logits[pos].argmax().item() != target)
                row = {
                    "phase": "probe_label",
                    "model_alias": model_alias("clean", seed),
                    "reference_clean_seed": seed,
                    "target": target,
                    "attack_target": target,
                    "sample_index": sample_index,
                    "eligible": eligible,
                    "pgd_status": "ineligible_original_target" if not eligible else ("censored" if not finite[pos] else "finite_label"),
                    "robustness_radius_pixels": None if not finite[pos] else float(radius[pos]),
                    "censored": None if not eligible else bool(not finite[pos]),
                    "original_prediction": int(logits[pos].argmax().item()),
                }
                row.update({f"feature_{name}": float(features[pos, col]) for col, name in enumerate(names)})
                probe_rows.append(row)
        probe = RidgeProbe.fit(np.asarray(feature_rows), np.asarray(label_rows), alpha=args.ridge_alpha, feature_names=names)
        probe.save(output / f"probe_target_{target}.npz")
        probes[target] = probe
        log(f"target={target}: Probe fitted on {len(label_rows)} finite reference records")
    write_csv(output / "probe_training_records.csv", probe_rows)
    write_json(output / "probe_parameters.json", {str(target): {"feature_names": list(probe.feature_names), "feature_mean": probe.feature_mean.tolist(), "feature_std": probe.feature_std.tolist(), "weight": probe.weight.tolist(), "bias": float(probe.bias), "alpha": probe.alpha} for target, probe in probes.items()})

    clean0, clean0_info = load_any_model(model_zoo_root, "clean", args.test_clean_seed, device=device)
    selected_by_target, selection_rows, score_rows = {}, [], []
    for target, probe in probes.items():
        logits, features, _ = logits_and_features(clean0, test_data, test_indices, target=target, batch_size=args.batch_size, device=device)
        predictions = logits.argmax(dim=1).numpy()
        scores = probe.predict(features)
        clean0_eligible = predictions != target
        # The Probe defines one shared Top-k set.  Do not remove Clean0
        # samples whose original prediction already equals the wrong target;
        # those samples are marked ineligible later for each model.
        positions = probe_topk_positions(scores, args.top_k).tolist()
        selected = [test_indices[pos] for pos in positions]
        selected_by_target[target] = selected
        for pos, index in enumerate(test_indices):
            score_rows.append({"target": target, "sample_index": index, "original_prediction": int(predictions[pos]), "probe_score": float(scores[pos]), "clean0_eligible": bool(clean0_eligible[pos])})
        for rank, pos in enumerate(positions, start=1):
            selection_rows.append({"target": target, "rank": rank, "sample_index": test_indices[pos], "original_prediction": int(predictions[pos]), "probe_score": float(scores[pos]), "selector": "probe_clean0_shared"})
    write_csv(output / "probe_selection_scores.csv", score_rows)
    write_csv(output / "selected_targeted_robust_samples.csv", selection_rows)

    explicit = {
        "ssba_encoder": Path(args.ssba_encoder_path) if args.ssba_encoder_path else None,
        "ssba_config": Path(args.ssba_config_path) if args.ssba_config_path else None,
        "inputaware": Path(args.inputaware_state_path) if args.inputaware_state_path else None,
        "adaptive_blend": Path(args.adaptive_blend_trigger_path) if args.adaptive_blend_trigger_path else None,
        "badnet": None, "blended": None, "wanet_identity": None, "wanet_noise": None,
    }
    adapters = build_trigger_adapters(
        model_root=artifact_root,
        backdoorbench_root=bdb_root,
        explicit=explicit,
        device=device,
        blended_alpha=args.blended_alpha,
        wanet_s=args.wanet_s,
        wanet_grid_rescale=args.wanet_grid_rescale,
    )
    analysis_groups = list(groups)
    model_cache = {"clean": clean0}
    model_info_rows = []
    for group in analysis_groups:
        model_cache[group], info = load_any_model(model_zoo_root, group, 0, device=device)
    quality_aliases = [
        model_alias("clean", seed)
        for seed in sorted(set((*reference_seeds, args.test_clean_seed)))
    ] + [model_alias(group, 0) for group in analysis_groups]
    for alias in quality_aliases:
        info = model_zoo_provenance["infos"][alias]
        model_info_rows.append({
            "model_alias": alias,
            "model_type": info.get("model_type"),
            "attack": info.get("attack"),
            "target_class": info.get("target_class"),
            "classifier_seed": info.get("classifier_seed"),
            "clean_acc": info.get("clean_acc"),
            "native_asr": info.get("asr"),
            "clean_trigger_asr": info.get("clean_trigger_asr"),
            "quality_gate": info.get("quality_gate"),
            "checkpoint": info.get("_artifact_paths", {}).get("checkpoint"),
            "config": json.dumps(info.get("config"), sort_keys=True, default=str),
        })

    attack_rows, trigger_rows, alignment_rows = [], [], []
    metric_rows = []
    trigger_prototypes = {}
    # Targeted PGD depends only on (model, wrong target), not on which
    # official trigger is being compared.  Cache it so adding six attack
    # families does not multiply the expensive PGD computation by six.
    attack_cache = {}
    all_model_aliases = ["clean", *analysis_groups]
    for target, selected_indices in selected_by_target.items():
        for trigger_type in analysis_groups:
            adapter = adapters[trigger_type]
            records = {}
            for internal_key in all_model_aliases:
                model = model_cache[internal_key]
                extractor = AvgPoolFeatures(model)
                try:
                    n = len(selected_indices)
                    cache_key = (internal_key, int(target))
                    if cache_key in attack_cache:
                        cached = attack_cache[cache_key]
                        base_features = cached["base_features"]
                        trigger_predictions = np.full(n, -1, dtype=np.int64)
                        trigger_success = np.full(n, False, dtype=bool)
                        trigger_features = None
                        trigger_delta = None
                        if adapter.status.available:
                            trigger_logits_parts, trigger_features_parts = [], []
                            for batch_indices, images, _ in batch_images(test_data, selected_indices, batch_size=args.batch_size, device=device):
                                triggered = adapter.apply(images, sample_indices=batch_indices, split="cifar100_test")
                                trigger_logits, trigger_features_batch = extractor(triggered)
                                trigger_logits_parts.append(trigger_logits.cpu().numpy())
                                trigger_features_parts.append(trigger_features_batch.cpu().numpy())
                            trigger_predictions = np.concatenate(trigger_logits_parts).argmax(axis=1)
                            trigger_success = trigger_predictions == 0
                            trigger_features = np.concatenate(trigger_features_parts)
                            trigger_delta = trigger_features - base_features
                        records[internal_key] = {
                            **cached,
                            "trigger_predictions": trigger_predictions,
                            "trigger_success": trigger_success,
                            "trigger_features": trigger_features,
                            "trigger_delta": trigger_delta,
                        }
                        continue
                    base_logits_parts, base_features_parts = [], []
                    endpoint_by_eps = {
                        eps: {"success": np.zeros(n, dtype=bool), "eligible": np.zeros(n, dtype=bool),
                              "prediction": np.zeros(n, dtype=np.int64), "linf": np.full(n, np.nan),
                              "loss": np.full(n, np.nan), "features": []}
                        for eps in analysis_eps
                    }
                    offset = 0
                    for batch_indices, images, _ in batch_images(test_data, selected_indices, batch_size=args.batch_size, device=device):
                        base_logits, base_features = extractor(images)
                        base_logits_np = base_logits.cpu().numpy()
                        base_features_np = base_features.cpu().numpy()
                        base_predictions = base_logits_np.argmax(axis=1)
                        base_logits_parts.append(base_logits_np)
                        base_features_parts.append(base_features_np)
                        local_count = len(batch_indices)
                        for eps in analysis_eps:
                            record = endpoint_by_eps[eps]
                            record["eligible"][offset:offset + local_count] = base_predictions != target
                            record["prediction"][offset:offset + local_count] = base_predictions
                            local_features = base_features_np.copy()
                            eligible_local = base_predictions != target
                            if np.any(eligible_local):
                                endpoint = targeted_pgd_endpoint(
                                    model,
                                    images[torch.as_tensor(eligible_local, device=images.device)],
                                    target=target,
                                    epsilon=eps / 255.0,
                                    steps=args.analysis_steps,
                                    alpha=eps / 255.0 / 10.0,
                                    random_start=True,
                                    restarts=args.analysis_restarts,
                                )
                                _, endpoint_features = extractor(endpoint.endpoint)
                                local_features[eligible_local] = endpoint_features.cpu().numpy()
                                positions = np.flatnonzero(eligible_local)
                                record["success"][offset + positions] = endpoint.success.cpu().numpy()
                                record["prediction"][offset + positions] = endpoint.endpoint_prediction.cpu().numpy()
                                record["linf"][offset + positions] = endpoint.endpoint_linf.cpu().numpy() * 255.0
                                record["loss"][offset + positions] = endpoint.target_loss.cpu().numpy()
                            record["features"].append(local_features)
                        offset += local_count
                    base_logits = np.concatenate(base_logits_parts)
                    base_features = np.concatenate(base_features_parts)
                    base_predictions = base_logits.argmax(axis=1)
                    for eps in analysis_eps:
                        endpoint_by_eps[eps]["features"] = np.concatenate(endpoint_by_eps[eps]["features"])

                    trigger_available = bool(adapter.status.available)
                    trigger_predictions = np.full(n, -1, dtype=np.int64)
                    trigger_success = np.full(n, False, dtype=bool)
                    trigger_features = None
                    trigger_delta = None
                    if trigger_available:
                        trigger_logits_parts, trigger_features_parts = [], []
                        for batch_indices, images, _ in batch_images(test_data, selected_indices, batch_size=args.batch_size, device=device):
                            triggered = adapter.apply(images, sample_indices=batch_indices, split="cifar100_test")
                            trigger_logits, trigger_features_batch = extractor(triggered)
                            trigger_logits_parts.append(trigger_logits.cpu().numpy())
                            trigger_features_parts.append(trigger_features_batch.cpu().numpy())
                        trigger_predictions = np.concatenate(trigger_logits_parts).argmax(axis=1)
                        trigger_success = trigger_predictions == 0
                        trigger_features = np.concatenate(trigger_features_parts)
                        trigger_delta = trigger_features - base_features
                    attack_cache[cache_key] = {
                        "base_logits": base_logits,
                        "base_features": base_features,
                        "base_predictions": base_predictions,
                        "eligible": base_predictions != target,
                        "endpoint_by_eps": endpoint_by_eps,
                    }
                    records[internal_key] = {
                        "base_logits": base_logits,
                        "base_features": base_features,
                        "base_predictions": base_predictions,
                        "eligible": base_predictions != target,
                        "endpoint_by_eps": endpoint_by_eps,
                        "trigger_predictions": trigger_predictions,
                        "trigger_success": trigger_success,
                        "trigger_features": trigger_features,
                        "trigger_delta": trigger_delta,
                    }
                finally:
                    extractor.close()

            bd_record = records[trigger_type]
            if adapter.status.available:
                cohort = backdoor_control_cohort(bd_record["eligible"], bd_record["trigger_success"])
            else:
                cohort = np.zeros(len(selected_indices), dtype=bool)
            cohort_id = f"bd_trigger_success_target{target}_{trigger_type}"
            for internal_key in all_model_aliases:
                record = records[internal_key]
                public_alias = model_alias("clean", args.test_clean_seed) if internal_key == "clean" else internal_key
                for pos, sample_index in enumerate(selected_indices):
                    trigger_rows.append({
                        "model_alias": public_alias,
                        "model_group": "clean" if internal_key == "clean" else internal_key,
                        "model_seed": 0,
                        "trigger_type": trigger_type,
                        "attack_target": target,
                        "true_backdoor_target": 0,
                        "sample_index": sample_index,
                        "trigger_source": adapter.status.source,
                        "trigger_available": adapter.status.available,
                        "alignment_available": adapter.status.available,
                        "alignment_unavailable_reason": adapter.status.reason if not adapter.status.available else None,
                        "original_prediction": int(record["base_predictions"][pos]),
                        "eligible": bool(record["eligible"][pos]),
                        "trigger_prediction": int(record["trigger_predictions"][pos]) if adapter.status.available else None,
                        "trigger_success_true_target": bool(record["trigger_success"][pos]) if adapter.status.available else None,
                        "control_cohort_id": cohort_id,
                        "in_control_cohort": bool(cohort[pos]),
                    })

                first_eps = np.full(n, np.inf, dtype=np.float64)
                for eps in analysis_eps:
                    success = record["endpoint_by_eps"][eps]["success"] & record["endpoint_by_eps"][eps]["eligible"]
                    first_eps[np.isinf(first_eps) & success] = eps
                target_resistance = target_margin(torch.from_numpy(record["base_logits"]), target).numpy()
                for eps in analysis_eps:
                    success = record["endpoint_by_eps"][eps]["success"] & record["endpoint_by_eps"][eps]["eligible"]
                    for pos, sample_index in enumerate(selected_indices):
                        eligible = bool(record["endpoint_by_eps"][eps]["eligible"][pos])
                        attack_rows.append({
                            "model_alias": public_alias,
                            "model_group": "clean" if internal_key == "clean" else internal_key,
                            "model_seed": 0,
                            "trigger_type": trigger_type,
                            "attack_target": target,
                            "true_backdoor_target": 0,
                            "sample_index": sample_index,
                            "epsilon_pixels": eps,
                            "before_prediction": int(record["base_predictions"][pos]),
                            "after_prediction": int(record["endpoint_by_eps"][eps]["prediction"][pos]) if eligible else None,
                            "original_prediction": int(record["base_predictions"][pos]),
                            "final_prediction": int(record["endpoint_by_eps"][eps]["prediction"][pos]) if eligible else None,
                            "eligible": eligible,
                            "pgd_status": "success" if eligible and bool(record["endpoint_by_eps"][eps]["success"][pos]) else ("failure" if eligible else "ineligible_original_target"),
                            "success": bool(record["endpoint_by_eps"][eps]["success"][pos]) if eligible else None,
                            "targeted_loss": float(record["endpoint_by_eps"][eps]["loss"][pos]) if eligible else None,
                            "actual_linf_pixels": float(record["endpoint_by_eps"][eps]["linf"][pos]) if eligible else None,
                            "first_success_epsilon_pixels": None if math.isinf(first_eps[pos]) else float(first_eps[pos]),
                            "censored": bool(eligible and math.isinf(first_eps[pos])),
                            "target_resistance_margin": float(target_resistance[pos]),
                            "steps": args.analysis_steps,
                            "random_start": True,
                            "restarts": args.analysis_restarts,
                            "control_cohort_id": cohort_id,
                            "in_control_cohort": bool(cohort[pos]),
                            "trigger_prediction": int(record["trigger_predictions"][pos]) if adapter.status.available else None,
                            "trigger_success_true_target": bool(record["trigger_success"][pos]) if adapter.status.available else None,
                            "alignment_available": bool(adapter.status.available),
                            "alignment_unavailable_reason": adapter.status.reason if not adapter.status.available else None,
                        })

                if not adapter.status.available:
                    for eps in analysis_eps:
                        for pos, sample_index in enumerate(selected_indices):
                            eligible = bool(record["endpoint_by_eps"][eps]["eligible"][pos])
                            alignment_rows.append({
                                "model_alias": public_alias,
                                "model_group": "clean" if internal_key == "clean" else internal_key,
                                "model_seed": 0,
                                "trigger_type": trigger_type,
                                "attack_target": target,
                                "true_backdoor_target": 0,
                                "sample_index": sample_index,
                                "epsilon_pixels": eps,
                                "protocol": f"fixed_{eps:g}",
                                "sample_group": "ineligible_original_target" if not eligible else "alignment_unavailable",
                                "eligible": eligible,
                                "success": bool(record["endpoint_by_eps"][eps]["success"][pos]) if eligible else None,
                                "control_cohort_id": cohort_id,
                                "in_control_cohort": bool(cohort[pos]),
                                "alignment_definition": "unavailable",
                                "same_alignment": None,
                                "shuffled_alignment": None,
                                "same_minus_shuffled": None,
                                "adversarial_feature_norm": None,
                                "trigger_feature_norm": None,
                                "trigger_success_true_target": None,
                                "trigger_concentration": None,
                                "adv_concentration": None,
                                "alignment_available": False,
                                "alignment_unavailable_reason": adapter.status.reason,
                            })
                    continue
                trigger_delta = record["trigger_delta"]
                trig_concentration = concentration(trigger_delta[cohort]) if np.any(cohort) else None
                trig_prototype = prototype(trigger_delta[cohort]) if np.any(cohort) else None
                if trig_prototype is not None:
                    trigger_prototypes[f"{trigger_type}:{public_alias}:target{target}"] = trig_prototype.tolist()
                for eps in analysis_eps:
                    endpoint = record["endpoint_by_eps"][eps]
                    adv_delta = endpoint["features"] - record["base_features"]
                    eligible = endpoint["eligible"]
                    success = endpoint["success"] & eligible
                    shared = trigger_type in SHARED_TRIGGER_GROUPS
                    alignment = cosine_to_vector(adv_delta, trig_prototype) if shared else cosine_rows(adv_delta, trigger_delta)
                    shuffled = np.full(n, np.nan)
                    if not shared:
                        shuffled = shuffled_alignment(adv_delta, trigger_delta, seed=2031 + target, repeats=args.shuffle_repeats)
                    for pos, sample_index in enumerate(selected_indices):
                        if not eligible[pos]:
                            sample_group = "ineligible_original_target"
                        elif not cohort[pos]:
                            sample_group = "outside_control_cohort"
                        else:
                            sample_group = "success" if success[pos] else "failure"
                        alignment_rows.append({
                            "model_alias": public_alias,
                            "model_group": "clean" if internal_key == "clean" else internal_key,
                            "model_seed": 0,
                            "trigger_type": trigger_type,
                            "attack_target": target,
                            "true_backdoor_target": 0,
                            "sample_index": sample_index,
                            "epsilon_pixels": eps,
                            "protocol": f"fixed_{eps:g}",
                            "sample_group": sample_group,
                            "eligible": bool(eligible[pos]),
                            "success": bool(success[pos]) if eligible[pos] else None,
                            "control_cohort_id": cohort_id,
                            "in_control_cohort": bool(cohort[pos]),
                            "alignment_definition": alignment_definition(trigger_type),
                            "same_alignment": float(alignment[pos]) if np.isfinite(alignment[pos]) and cohort[pos] and eligible[pos] else None,
                            "shuffled_alignment": float(shuffled[pos]) if np.isfinite(shuffled[pos]) and cohort[pos] and eligible[pos] and not shared else None,
                            "same_minus_shuffled": float(alignment[pos] - shuffled[pos]) if np.isfinite(alignment[pos]) and np.isfinite(shuffled[pos]) and cohort[pos] and eligible[pos] and not shared else None,
                            "adversarial_feature_norm": float(np.linalg.norm(adv_delta[pos])) if eligible[pos] else None,
                            "trigger_feature_norm": float(np.linalg.norm(trigger_delta[pos])),
                            "trigger_success_true_target": bool(record["trigger_success"][pos]),
                            "trigger_concentration": trig_concentration,
                            "adv_concentration": concentration(adv_delta[cohort & success]) if np.any(cohort & success) else None,
                            "alignment_available": True,
                            "alignment_unavailable_reason": None,
                        })

            log(f"target={target}, trigger={trigger_type}: paired cohort size={int(cohort.sum())}")

    grouped = defaultdict(list)
    for row in alignment_rows:
        grouped[(row["target"], row["trigger_type"], row["model_alias"], row["epsilon_pixels"])].append(row)
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        target, trigger_type, alias, epsilon = key
        cohort_rows = [row for row in rows if row["in_control_cohort"] and row["eligible"]]
        success_rows = [row for row in cohort_rows if row["success"] is True]
        failure_rows = [row for row in cohort_rows if row["success"] is False]
        alignments = [float(row["same_alignment"]) for row in success_rows if row["same_alignment"] is not None]
        failure_alignments = [float(row["same_alignment"]) for row in failure_rows if row["same_alignment"] is not None]
        shuffled = [float(row["shuffled_alignment"]) for row in success_rows if row["shuffled_alignment"] is not None]
        failure_shuffled = [float(row["shuffled_alignment"]) for row in failure_rows if row["shuffled_alignment"] is not None]
        metric_rows.append({
            "target": target,
            "trigger_type": trigger_type,
            "model_alias": alias,
            "epsilon_pixels": epsilon,
            "metric": "alignment_success",
            "value": float(np.mean(alignments)) if alignments else None,
            "n": len(success_rows),
            "alignment_definition": rows[0]["alignment_definition"],
        })
        metric_rows.append({
            "target": target,
            "trigger_type": trigger_type,
            "model_alias": alias,
            "epsilon_pixels": epsilon,
            "metric": "alignment_failure",
            "value": float(np.mean(failure_alignments)) if failure_alignments else None,
            "n": len(failure_rows),
            "alignment_definition": rows[0]["alignment_definition"],
        })
        metric_rows.append({
            "target": target,
            "trigger_type": trigger_type,
            "model_alias": alias,
            "epsilon_pixels": epsilon,
            "metric": "adv_concentration_success",
            "value": rows[0]["adv_concentration"] if rows else None,
            "n": len(success_rows),
            "alignment_definition": rows[0]["alignment_definition"],
        })
        if trigger_type in SHARED_TRIGGER_GROUPS:
            metric_rows.append({
                "target": target,
                "trigger_type": trigger_type,
                "model_alias": alias,
                "epsilon_pixels": epsilon,
                "metric": "trigger_concentration",
                "value": rows[0]["trigger_concentration"] if rows else None,
                "n": len(cohort_rows),
                "alignment_definition": "shared_trigger",
            })
        if trigger_type in SAMPLE_TRIGGER_GROUPS:
            metric_rows.append({
                "target": target,
                "trigger_type": trigger_type,
                "model_alias": alias,
                "epsilon_pixels": epsilon,
                "metric": "same_minus_shuffle_success",
                "value": float(np.mean(alignments) - np.mean(shuffled)) if alignments and shuffled else None,
                "n": len(success_rows),
                "alignment_definition": "same_pair_vs_shuffle",
            })
            metric_rows.append({
                "target": target,
                "trigger_type": trigger_type,
                "model_alias": alias,
                "epsilon_pixels": epsilon,
                "metric": "same_minus_shuffle_failure",
                "value": float(np.mean(failure_alignments) - np.mean(failure_shuffled)) if failure_alignments and failure_shuffled else None,
                "n": len(failure_rows),
                "alignment_definition": "same_pair_vs_shuffle",
            })

    quality_rows = model_info_rows
    clean_public_alias = model_alias("clean", args.test_clean_seed)
    main_comparisons = []
    for target in targets:
        for trigger_type in analysis_groups:
            for epsilon in analysis_eps:
                rows = [row for row in alignment_rows if row["target"] == target and row["trigger_type"] == trigger_type and row["epsilon_pixels"] == epsilon]
                by_alias = defaultdict(list)
                for row in rows:
                    by_alias[row["model_alias"]].append(row)
                bd_rows = [row for row in by_alias.get(model_alias(trigger_type, 0), []) if row["in_control_cohort"] and row["eligible"] and row["success"] is True]
                clean_rows = [row for row in by_alias.get(clean_public_alias, []) if row["in_control_cohort"] and row["eligible"] and row["success"] is True]
                bd_alignment = [float(row["same_alignment"]) for row in bd_rows if row["same_alignment"] is not None]
                clean_alignment = [float(row["same_alignment"]) for row in clean_rows if row["same_alignment"] is not None]
                bd_failure_rows = [row for row in by_alias.get(model_alias(trigger_type, 0), []) if row["in_control_cohort"] and row["eligible"] and row["success"] is False]
                bd_failure_alignment = [float(row["same_alignment"]) for row in bd_failure_rows if row["same_alignment"] is not None]
                bd_shuffled = [float(row["shuffled_alignment"]) for row in bd_rows if row["shuffled_alignment"] is not None]
                clean_shuffled = [float(row["shuffled_alignment"]) for row in clean_rows if row["shuffled_alignment"] is not None]
                comparison = {
                    "target": target,
                    "trigger_type": trigger_type,
                    "epsilon_pixels": epsilon,
                    "backdoor_alias": model_alias(trigger_type, 0),
                    "clean_alias": clean_public_alias,
                    "alignment_bd_success": float(np.mean(bd_alignment)) if bd_alignment else None,
                    "alignment_clean_success": float(np.mean(clean_alignment)) if clean_alignment else None,
                    "alignment_bd_gt_clean": bool(bd_alignment and clean_alignment and np.mean(bd_alignment) > np.mean(clean_alignment)),
                    "alignment_bd_failure": float(np.mean(bd_failure_alignment)) if bd_failure_alignment else None,
                    "alignment_bd_gt_failure": bool(bd_alignment and bd_failure_alignment and np.mean(bd_alignment) > np.mean(bd_failure_alignment)) if trigger_type in SAMPLE_TRIGGER_GROUPS else None,
                    "alignment_bd_same": float(np.mean(bd_alignment)) if bd_alignment and trigger_type in SAMPLE_TRIGGER_GROUPS else None,
                    "alignment_bd_shuffle": float(np.mean(bd_shuffled)) if bd_shuffled and trigger_type in SAMPLE_TRIGGER_GROUPS else None,
                    "alignment_clean_same": float(np.mean(clean_alignment)) if clean_alignment and trigger_type in SAMPLE_TRIGGER_GROUPS else None,
                    "alignment_clean_shuffle": float(np.mean(clean_shuffled)) if clean_shuffled and trigger_type in SAMPLE_TRIGGER_GROUPS else None,
                    "alignment_bd_same_gt_shuffle": bool(bd_alignment and bd_shuffled and np.mean(bd_alignment) > np.mean(bd_shuffled)) if trigger_type in SAMPLE_TRIGGER_GROUPS else None,
                    "c_adv_bd_success": bd_rows[0]["adv_concentration"] if bd_rows else None,
                    "c_adv_clean_success": clean_rows[0]["adv_concentration"] if clean_rows else None,
                    "c_adv_bd_gt_clean": bool(bd_rows and clean_rows and bd_rows[0]["adv_concentration"] is not None and clean_rows[0]["adv_concentration"] is not None and bd_rows[0]["adv_concentration"] > clean_rows[0]["adv_concentration"]),
                    "control_cohort_n": len([row for row in rows if row["in_control_cohort"]]),
                    "alignment_definition": rows[0]["alignment_definition"] if rows else None,
                }
                if trigger_type in SHARED_TRIGGER_GROUPS:
                    bd_trigger = [row["trigger_concentration"] for row in by_alias.get(model_alias(trigger_type, 0), []) if row["in_control_cohort"] and row["trigger_concentration"] is not None]
                    clean_trigger = [row["trigger_concentration"] for row in by_alias.get(clean_public_alias, []) if row["in_control_cohort"] and row["trigger_concentration"] is not None]
                    comparison.update({
                        "c_trig_bd": bd_trigger[0] if bd_trigger else None,
                        "c_trig_clean": clean_trigger[0] if clean_trigger else None,
                        "c_trig_bd_gt_clean": bool(bd_trigger and clean_trigger and bd_trigger[0] > clean_trigger[0]),
                    })
                else:
                    comparison.update({"c_trig_bd": None, "c_trig_clean": None, "c_trig_bd_gt_clean": None})
                main_comparisons.append(comparison)
    write_csv(output / "trigger_records.csv", trigger_rows)
    write_csv(output / "attack_records.csv", attack_rows)
    write_csv(output / "feature_alignment_records.csv", alignment_rows)
    write_csv(output / "group_metrics.csv", metric_rows)
    write_csv(output / "model_quality.csv", quality_rows)
    write_figures(output, alignment_rows)
    summary = {
        **config,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "selected_indices_by_target": selected_by_target,
        "trigger_status": {group: vars(adapter.status) for group, adapter in adapters.items()},
        "trigger_prototypes": trigger_prototypes,
        "analysis_backdoor_groups": analysis_groups,
        "model_quality": quality_rows,
        "model_zoo_full_info": model_zoo_provenance["infos"],
        "model_zoo_package_version": model_zoo_provenance["package_version"],
        "model_zoo_registry_sha256": model_zoo_provenance["registry_sha256"],
        "model_zoo_registered_aliases": model_zoo_provenance["registered_aliases"],
        "main_comparisons": main_comparisons,
        "ssba_provenance_note": (
            "SSBA alignment uses the official encoder and provenance configuration. "
            "The historical CIFAR-10 replacement-array comparison may differ by a "
            "negligible uint8-level numerical/quantization amount (maximum one gray "
            "level); this does not disable encoder-based SSBA trigger analysis."
        ),
        "outputs": {
            "attack_records": str((output / "attack_records.csv").resolve()),
            "alignment_records": str((output / "feature_alignment_records.csv").resolve()),
            "main_comparisons": str((output / "main_comparisons.json").resolve()),
        },
        "interpretation": "wrong-target attack to official-trigger-related feature direction; not a literal A->B->C class-transition claim",
    }
    write_json(output / "main_comparisons.json", main_comparisons)
    write_json(output / "summary.json", summary)
    log(f"Stage 1D-WT complete: {output.resolve()}")


if __name__ == "__main__":
    main()
