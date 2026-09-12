from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from trigger_alignment_wrong_target import (
    AvgPoolFeatures,
    backdoor_control_cohort,
    alignment_definition,
    model_alias,
    parse_floats,
    probe_topk_positions,
)
from mdluap.official_triggers import _array, _quantize_like_torchvision_save_image


def test_model_aliases_are_explicit():
    assert model_alias("clean", 0) == "clean0"
    assert model_alias("clean", 3) == "clean3"
    assert model_alias("adaptive_blend", 0) == "adaptive_blend01"
    assert model_alias("ssba", 0) == "ssba0"


def test_control_cohort_uses_backdoor_only():
    eligible = np.array([True, True, False, True])
    trigger_success = np.array([True, False, True, True])
    np.testing.assert_array_equal(backdoor_control_cohort(eligible, trigger_success), [True, False, False, True])


def test_ineligible_samples_are_not_successes():
    eligible = np.array([False, True])
    success = np.array([False, True])
    assert not bool((success & eligible)[0])
    assert bool((success & eligible)[1])


def test_probe_topk_does_not_reselect_for_clean0_eligibility():
    scores = np.array([10.0, 9.0, 8.0])
    # The highest-scoring sample may be ineligible for one model; selection
    # remains Probe-defined and is not recomputed from model predictions.
    np.testing.assert_array_equal(probe_topk_positions(scores, 2), [0, 1])


def test_trigger_alignment_branches_are_not_universal_same_shuffle():
    assert alignment_definition("badnet") == "prototype"
    assert alignment_definition("adaptive_blend") == "prototype"
    assert alignment_definition("ssba") == "same_pair"
    assert alignment_definition("inputaware") == "same_pair"


def test_analysis_eps_parser_accepts_two_budgets():
    assert parse_floats("1,1.5") == (1.0, 1.5)


def test_feature_extractor_supports_layer4_fallback():
    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Conv2d(3, 4, 1)
            self.fc = nn.Linear(4, 10)

        def forward(self, x):
            value = self.layer4(x).mean(dim=(2, 3))
            return self.fc(value)

    extractor = AvgPoolFeatures(TinyClassifier())
    try:
        logits, features = extractor(torch.zeros(2, 3, 8, 8))
    finally:
        extractor.close()
    assert logits.shape == (2, 10)
    assert features.shape == (2, 4)


def test_ssba_array_accepts_hwc(tmp_path):
    path = tmp_path / "ssba.npy"
    np.save(path, np.zeros((2, 32, 32, 3), dtype=np.uint8))
    values = _array(path)
    assert tuple(values.shape) == (2, 3, 32, 32)


def test_ssba_quantization_matches_torchvision_half_up():
    # torchvision.save_image() maps half-integer uint8 levels using
    # floor(255 * value + 0.5), whereas torch.round() uses round-to-even.
    values = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32) / 255.0
    quantized = _quantize_like_torchvision_save_image(values) * 255.0
    torch.testing.assert_close(quantized, torch.tensor([1.0, 2.0, 3.0]))
