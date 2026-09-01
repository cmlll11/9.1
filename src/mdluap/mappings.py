"""The official image-dependent x+f GAP mapping used by this experiment."""

from __future__ import annotations

import torch
from torch import nn


class ImageDependentResidualMapping(nn.Module):
    """Apply the official GAP generator as a bounded residual mapping."""

    def __init__(self, generator: nn.Module, epsilon: float):
        super().__init__()
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        self.generator = generator
        self.epsilon = float(epsilon)

    def effective_epsilon(self) -> torch.Tensor:
        """Return the fixed raw-pixel perturbation bound as a tensor."""

        parameter = next(self.generator.parameters())
        return parameter.new_tensor(self.epsilon)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Generate x+f(x), clip the residual, and keep pixels in [0, 1]."""

        raw_delta = self.generator(images)
        delta = raw_delta.clamp(-1.0, 1.0) * self.effective_epsilon().to(images.dtype)
        return (images + delta).clamp(0.0, 1.0)
