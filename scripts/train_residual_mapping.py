"""Train one separate image-dependent x+f(x) GAP mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mdluap.data import cifar10_split, loader
from mdluap.gap import build_official_gap_generator, seed_everything, train_residual_gap
from mdluap.mappings import ImageDependentResidualMapping
from mdluap.models import load_attack_result_model


def parse_fraction(value: str) -> float:
    """Accept a decimal or a readable fraction such as 16/255."""

    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--gap-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attack-goal", choices=("targeted", "non_targeted"), required=True)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--epsilon", type=parse_fraction, default=parse_fraction("16/255"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model, result_meta = load_attack_result_model(
        args.result,
        backdoorbench_root=args.backdoorbench_root,
        device=device,
    )
    train_set = cifar10_split(args.data_root, split="train", split_seed=args.split_seed)
    train_loader = loader(train_set, batch_size=args.batch_size, train=True, workers=args.workers)

    # This branch has exactly three generator outputs: f(x) is a residual.
    generator = build_official_gap_generator(
        gap_root=args.gap_root,
        device=device,
        ngf=args.ngf,
        output_channels=3,
    )
    mapping = ImageDependentResidualMapping(generator, args.epsilon).to(device)
    training = train_residual_gap(
        model=model,
        mapping=mapping,
        train_loader=train_loader,
        attack_goal=args.attack_goal,
        target_label=args.target,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        checkpoint_path=str(output / "checkpoint.pt"),
        max_batches=args.max_batches,
    )

    final_path = output / "mapping.pt"
    final_tmp = output / "mapping.pt.tmp"
    torch.save(
        {
            "protocol": "MDL-UAP-residual-v1",
            "mode": "imdep_residual",
            "mapping_form": "x+f(x)",
            "attack_goal": args.attack_goal,
            "target": args.target,
            "epsilon": args.epsilon,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "max_batches": args.max_batches,
            "ngf": args.ngf,
            "mapping": mapping.state_dict(),
            "training_history": training["history"],
            "result": str(Path(args.result).resolve()),
            "model_name": result_meta["model_name"],
        },
        final_tmp,
    )
    final_tmp.replace(final_path)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Run complete: output={output.resolve()}")


if __name__ == "__main__":
    main()
