"""Package an official BackdoorBench clean_model.pth for the UAP loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="preactresnet18")
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {"model_name": args.model_name, "num_classes": args.num_classes, "model": state, "source": args.weights},
        temporary,
    )
    temporary.replace(output)
    print(f"Run complete: output={output.resolve()}")


if __name__ == "__main__":
    main()
