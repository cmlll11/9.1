"""Create disjoint CIFAR-10 train partitions for the hard-sample experiment."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def load_train(root: Path) -> tuple[np.ndarray, list[int]]:
    batch_root = root / "cifar10" / "cifar-10-batches-py"
    arrays = []
    labels: list[int] = []
    for index in range(1, 6):
        with (batch_root / f"data_batch_{index}").open("rb") as handle:
            item = pickle.load(handle, encoding="bytes")
        arrays.append(np.asarray(item[b"data"], dtype=np.uint8))
        labels.extend(int(value) for value in item[b"labels"])
    return np.concatenate(arrays, axis=0), labels


def write_partition(root: Path, data: np.ndarray, labels: list[int], meta: dict) -> None:
    batch_root = root / "cifar10" / "cifar-10-batches-py"
    batch_root.mkdir(parents=True, exist_ok=True)
    boundaries = np.array_split(np.arange(len(labels)), 5)
    for batch_index, indices in enumerate(boundaries, start=1):
        if len(indices) == 0:
            raise ValueError("each partition batch must contain at least one sample")
        # torchvision loads CIFAR-10 pickle keys as strings with latin1.
        payload = {
            "batch_label": f"hard-sample partition {batch_index}",
            "data": data[indices],
            "labels": [labels[int(i)] for i in indices],
            "filenames": [f"partition_{int(i)}.png" for i in indices],
        }
        with (batch_root / f"data_batch_{batch_index}").open("wb") as handle:
            pickle.dump(payload, handle, protocol=2)
    with (batch_root / "batches.meta").open("wb") as handle:
        pickle.dump(
            {
                "num_cases_per_batch": 10000,
                "label_names": [str(i) for i in range(10)],
                "num_vis": 3072,
            },
            handle,
            protocol=2,
        )
    source_test = meta.get("source_test_batch")
    if source_test:
        shutil.copy2(source_test, batch_root / "test_batch")
    (root / "partition.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = Path(args.data_root)
    data, labels = load_train(source)
    if args.selection_size <= 0 or args.selection_size >= len(labels):
        raise ValueError("selection-size must be between 1 and the full train size")
    order = np.random.default_rng(args.seed).permutation(len(labels))
    selection_indices = order[: args.selection_size].tolist()
    shared_indices = order[args.selection_size :].tolist()
    if set(selection_indices).intersection(shared_indices):
        raise RuntimeError("partition intersection is not empty")

    output = Path(args.output_root)
    write_partition(
        output / "selection",
        data[selection_indices],
        [labels[i] for i in selection_indices],
        {"partition": "selection", "seed": args.seed, "size": len(selection_indices), "indices": selection_indices, "source_test_batch": str((source / "cifar10" / "cifar-10-batches-py" / "test_batch").resolve())},
    )
    write_partition(
        output / "shared",
        data[shared_indices],
        [labels[i] for i in shared_indices],
        {"partition": "shared", "seed": args.seed, "size": len(shared_indices), "indices": shared_indices, "source_test_batch": str((source / "cifar10" / "cifar-10-batches-py" / "test_batch").resolve())},
    )
    manifest = {
        "protocol": "hard-sample-cifar10-partitions-v1",
        "seed": args.seed,
        "selection_size": len(selection_indices),
        "shared_size": len(shared_indices),
        "intersection_size": 0,
        "selection": str((output / "selection").resolve()),
        "shared": str((output / "shared").resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Partitions complete: selection={len(selection_indices)} shared={len(shared_indices)}")


if __name__ == "__main__":
    main()
