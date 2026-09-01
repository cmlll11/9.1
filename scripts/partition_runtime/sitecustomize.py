"""Allow torchvision to read the intentionally rebatched experiment data."""

from __future__ import annotations

import os
import pickle

from torchvision.datasets import CIFAR10


_original_check_integrity = CIFAR10._check_integrity
_original_load_meta = CIFAR10._load_meta


def _check_integrity(self) -> bool:
    root = os.path.abspath(str(self.root))
    if "hard_sample_gap" not in root:
        return _original_check_integrity(self)
    required = [filename for filename, _ in self.train_list + self.test_list]
    return all(os.path.isfile(os.path.join(self.root, self.base_folder, name)) for name in required)


CIFAR10._check_integrity = _check_integrity


def _load_meta(self) -> None:
    """Load the rebatched metadata without checking the official MD5."""
    root = os.path.abspath(str(self.root))
    if "hard_sample_gap" not in root:
        _original_load_meta(self)
        return
    path = os.path.join(self.root, self.base_folder, self.meta["filename"])
    with open(path, "rb") as infile:
        data = pickle.load(infile, encoding="latin1")
    self.classes = data[self.meta["key"]]
    self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


CIFAR10._load_meta = _load_meta
