"""Deterministic, grid-based targeted PGD attacks for the pilot experiment.

The functions in this module operate on raw images in ``[0, 1]``.  The model
passed by the caller is expected to perform its own input normalization, as
``mdluap.models.NormalizedClassifier`` does for BackdoorBench checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# The original pilot grid is kept unchanged for reproducibility of Stage 1A/1B.
EPSILON_GRID: tuple[float, ...] = tuple(value / 255.0 for value in (1, 2, 4, 8, 16, 32))

# The transfer experiment resolves the informative low-radius region more
# finely.  Values are stored in raw-image units even though the protocol is
# reported in pixel units divided by 255.
FINE_EPSILON_GRID: tuple[float, ...] = tuple(
    value / 255.0 for value in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
)

# Target-free Stage 1C uses a finer low-radius grid and keeps 16/255 and
# 32/255 available so that highly robust training examples are less likely to
# be right-censored before the Ridge Probe is fitted.
UNTARGETED_EPSILON_GRID: tuple[float, ...] = tuple(
    value / 255.0
    for value in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0)
)


@dataclass
class PGDResult:
    """Per-image result of a targeted PGD run at one epsilon budget."""

    success: Tensor
    best_linf: Tensor
    best_prediction: Tensor


@dataclass
class TargetedEndpointResult:
    """Targeted-PGD endpoint selected by the lowest target loss.

    ``success`` records whether any iterate in any restart reached the target
    class.  ``endpoint`` is always populated: it is the iterate with the
    lowest target cross-entropy, including for failed attacks.  Keeping an
    endpoint for failures makes feature-direction comparisons use the same
    representation for success and failure groups.
    """

    endpoint: Tensor
    success: Tensor
    endpoint_linf: Tensor
    endpoint_prediction: Tensor
    target_loss: Tensor


@dataclass
class GridRadiusResult:
    """First-success epsilon on a discrete grid, including right censoring."""

    radius: Tensor
    success: Tensor
    censored: Tensor


def untargeted_pgd(
    model: nn.Module,
    images: Tensor,
    *,
    epsilon: float,
    steps: int,
    alpha: float,
    random_start: bool,
    restarts: int,
) -> PGDResult:
    """Run an untargeted :math:`L_\infty` PGD attack.

    The original model prediction is computed once before optimization and
    is then used as a frozen pseudo-label.  The attack succeeds when the
    prediction of the perturbed image differs from that original prediction.
    The objective maximizes cross-entropy with respect to the frozen original
    label, so the signed gradient update uses ``+alpha``.

    Parameters
    ----------
    model:
        Classifier accepting raw images with shape ``[N, 3, H, W]`` and values
        in ``[0, 1]``.  The model is expected to apply its own normalization.
    images:
        Unmodified raw input tensor with shape ``[N, 3, H, W]`` in ``[0, 1]``.
    epsilon:
        Maximum per-pixel perturbation in raw image units.
    steps, alpha:
        Number of signed-gradient updates and update size in raw image units.
    random_start, restarts:
        Whether each restart begins uniformly inside the Linf ball and the
        number of independent attack attempts.

    Returns
    -------
    PGDResult
        ``success`` and ``best_linf`` have shape ``[N]``.  ``best_linf`` is
        ``inf`` when no restart changes the prediction.  ``best_prediction``
        stores the prediction associated with the smallest successful norm,
        or the original prediction when unsuccessful.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected images with shape [N, 3, H, W], got {tuple(images.shape)}")
    if not 0.0 < float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if int(steps) <= 0 or int(restarts) <= 0:
        raise ValueError("steps and restarts must be positive")

    model.eval()
    with torch.no_grad():
        original_prediction = model(images).argmax(dim=1)

    best_linf = torch.full((images.shape[0],), float("inf"), device=images.device)
    best_prediction = original_prediction.clone()

    for _ in range(int(restarts)):
        if random_start:
            delta = torch.empty_like(images).uniform_(-float(epsilon), float(epsilon))
            delta = _project_delta(images, delta, epsilon)
        else:
            delta = torch.zeros_like(images)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            logits = model((images + delta).clamp(0.0, 1.0))
            # Untargeted PGD maximizes loss for the frozen original label.
            loss = F.cross_entropy(logits, original_prediction)
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            with torch.no_grad():
                delta = _project_delta(images, delta + float(alpha) * gradient.sign(), epsilon)
                predictions = model((images + delta).clamp(0.0, 1.0)).argmax(dim=1)
                success = predictions.ne(original_prediction)
                current_linf = delta.abs().flatten(1).amax(dim=1)
                update = success & (current_linf < best_linf)
                best_linf = torch.where(update, current_linf, best_linf)
                best_prediction = torch.where(update, predictions, best_prediction)

    return PGDResult(
        success=torch.isfinite(best_linf),
        best_linf=best_linf,
        best_prediction=best_prediction,
    )


def estimate_untargeted_grid_radius(
    model: nn.Module,
    images: Tensor,
    *,
    epsilons: Iterable[float] = UNTARGETED_EPSILON_GRID,
    steps: int,
    alpha_fraction: float = 0.1,
    random_start: bool = False,
    restarts: int = 1,
) -> GridRadiusResult:
    """Estimate untargeted robustness by the first successful epsilon grid.

    A sample is right-censored when no tested epsilon changes its original
    prediction.  Its radius is therefore strictly larger than the largest
    tested budget and must not be replaced by that endpoint in means or Ridge
    labels.
    """

    epsilon_values = tuple(float(value) for value in epsilons)
    if not epsilon_values or tuple(sorted(epsilon_values)) != epsilon_values:
        raise ValueError("epsilons must be a non-empty ascending sequence")

    n = images.shape[0]
    radius = torch.full((n,), float("inf"), device=images.device)
    success = torch.zeros((n,), dtype=torch.bool, device=images.device)
    for epsilon in epsilon_values:
        result = untargeted_pgd(
            model,
            images,
            epsilon=epsilon,
            steps=steps,
            alpha=epsilon * float(alpha_fraction),
            random_start=random_start,
            restarts=restarts,
        )
        newly_successful = (~success) & result.success
        radius = torch.where(newly_successful, torch.full_like(radius, epsilon), radius)
        success = success | result.success

    return GridRadiusResult(radius=radius, success=success, censored=~success)


def _target_tensor(images: Tensor, target: int) -> Tensor:
    """Create one target label for every image in a batch."""

    return torch.full((images.shape[0],), int(target), dtype=torch.long, device=images.device)


def _project_delta(images: Tensor, delta: Tensor, epsilon: float) -> Tensor:
    """Project a raw-image perturbation into both valid pixel and Linf boxes."""

    delta = delta.clamp(-float(epsilon), float(epsilon))
    # Clamping the image can make the effective perturbation smaller than the
    # requested delta.  Recompute it so the recorded Linf norm is exact.
    return (images + delta).clamp(0.0, 1.0) - images


def targeted_pgd(
    model: nn.Module,
    images: Tensor,
    *,
    target: int,
    epsilon: float,
    steps: int,
    alpha: float,
    random_start: bool,
    restarts: int,
) -> PGDResult:
    """Run targeted Linf PGD and retain the smallest successful perturbation.

    Parameters
    ----------
    model:
        A classifier accepting ``[N, 3, H, W]`` raw images in ``[0, 1]``.
    images:
        A batch of raw images with shape ``[N, C, H, W]`` and values in
        ``[0, 1]``.  The tensor is not modified in-place.
    target, epsilon, steps, alpha:
        Target class, Linf budget, iteration count, and signed-gradient step
        size.  All epsilon and alpha values use raw pixel units.
    random_start, restarts:
        Whether to initialize each restart uniformly inside the Linf ball and
        how many independent runs to perform.  The caller uses one zero-start
        run for coarse search and three random starts for refinement.

    Returns
    -------
    PGDResult
        ``success`` and ``best_linf`` have shape ``[N]``.  ``best_linf`` is
        ``inf`` for images for which no restart found the target class.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected images with shape [N, 3, H, W], got {tuple(images.shape)}")
    if not 0.0 < float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if int(steps) <= 0 or int(restarts) <= 0:
        raise ValueError("steps and restarts must be positive")

    model.eval()
    targets = _target_tensor(images, target)
    best_linf = torch.full((images.shape[0],), float("inf"), device=images.device)
    best_prediction = model(images).argmax(dim=1)

    for restart in range(int(restarts)):
        if random_start:
            delta = torch.empty_like(images).uniform_(-float(epsilon), float(epsilon))
            delta = _project_delta(images, delta, epsilon)
        else:
            delta = torch.zeros_like(images)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            logits = model((images + delta).clamp(0.0, 1.0))
            loss = F.cross_entropy(logits, targets)
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            # Targeted PGD minimizes the target loss, hence the negative sign.
            with torch.no_grad():
                delta = _project_delta(images, delta - float(alpha) * gradient.sign(), epsilon)

            with torch.no_grad():
                predictions = model((images + delta).clamp(0.0, 1.0)).argmax(dim=1)
                success = predictions.eq(targets)
                current_linf = delta.abs().flatten(1).amax(dim=1)
                update = success & (current_linf < best_linf)
                best_linf = torch.where(update, current_linf, best_linf)
                best_prediction = torch.where(update, predictions, best_prediction)

        # A zero-step success is relevant for epsilon=0 accounting, but the
        # pilot excludes such images from robust-sample ranking.
        if restart == 0:
            with torch.no_grad():
                initial_success = model(images).argmax(dim=1).eq(targets)
                best_linf = torch.where(initial_success, torch.zeros_like(best_linf), best_linf)
                best_prediction = torch.where(initial_success, targets, best_prediction)

    return PGDResult(
        success=torch.isfinite(best_linf),
        best_linf=best_linf,
        best_prediction=best_prediction,
    )


def targeted_pgd_endpoint(
    model: nn.Module,
    images: Tensor,
    *,
    target: int,
    epsilon: float,
    steps: int,
    alpha: float,
    random_start: bool,
    restarts: int,
) -> TargetedEndpointResult:
    """Run targeted PGD and retain one comparable endpoint per image.

    Unlike :func:`targeted_pgd`, this function returns an adversarial image for
    every input, including failed attacks.  The returned endpoint is the
    iterate with the smallest target cross-entropy over all steps and
    restarts.  ``success`` is tracked independently as ``True`` when any
    iterate was predicted as ``target``.  This separation is intentional:
    the Stage 1D mechanism experiment compares feature changes for successful
    and failed attacks at the same epsilon grid without dropping failures.

    Inputs are raw ``[N, 3, H, W]`` images in ``[0, 1]``.  Epsilon and alpha
    are also expressed in raw pixel units.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected images with shape [N, 3, H, W], got {tuple(images.shape)}")
    if not 0.0 < float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    if int(steps) <= 0 or int(restarts) <= 0:
        raise ValueError("steps and restarts must be positive")

    model.eval()
    targets = _target_tensor(images, target)
    n = images.shape[0]
    best_loss = torch.full((n,), float("inf"), device=images.device)
    best_endpoint = images.detach().clone()
    best_linf = torch.zeros((n,), device=images.device)
    best_prediction = model(images).argmax(dim=1)
    attack_success = best_prediction.eq(targets)

    for _ in range(int(restarts)):
        if random_start:
            delta = torch.empty_like(images).uniform_(-float(epsilon), float(epsilon))
            delta = _project_delta(images, delta, epsilon)
        else:
            delta = torch.zeros_like(images)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            adversarial = (images + delta).clamp(0.0, 1.0)
            logits = model(adversarial)
            losses = F.cross_entropy(logits, targets, reduction="none")
            gradient = torch.autograd.grad(losses.sum(), delta, only_inputs=True)[0]

            with torch.no_grad():
                # Targeted PGD minimizes target cross-entropy.
                delta = _project_delta(images, delta - float(alpha) * gradient.sign(), epsilon)
                endpoint = (images + delta).clamp(0.0, 1.0)
                endpoint_logits = model(endpoint)
                endpoint_losses = F.cross_entropy(endpoint_logits, targets, reduction="none")
                endpoint_predictions = endpoint_logits.argmax(dim=1)
                attack_success |= endpoint_predictions.eq(targets)
                better = endpoint_losses < best_loss
                best_loss = torch.where(better, endpoint_losses, best_loss)
                current_linf = delta.abs().flatten(1).amax(dim=1)
                best_linf = torch.where(better, current_linf, best_linf)
                best_prediction = torch.where(better, endpoint_predictions, best_prediction)
                best_endpoint = torch.where(better.view(-1, 1, 1, 1), endpoint, best_endpoint)

    return TargetedEndpointResult(
        endpoint=best_endpoint.detach(),
        success=attack_success.detach(),
        endpoint_linf=best_linf.detach(),
        endpoint_prediction=best_prediction.detach(),
        target_loss=best_loss.detach(),
    )


def estimate_grid_radius(
    model: nn.Module,
    images: Tensor,
    *,
    target: int,
    epsilons: Iterable[float] = EPSILON_GRID,
    steps: int,
    alpha_fraction: float = 0.1,
    random_start: bool = False,
    restarts: int = 1,
) -> GridRadiusResult:
    """Estimate target-specific radius by the first successful epsilon grid.

    The returned ``censored`` flag is true when no attack succeeds at any grid
    point.  Such a sample has radius strictly larger than the largest tested
    epsilon and must not be replaced by that endpoint in downstream means.
    """

    epsilon_values = tuple(float(value) for value in epsilons)
    if not epsilon_values or tuple(sorted(epsilon_values)) != epsilon_values:
        raise ValueError("epsilons must be a non-empty ascending sequence")

    n = images.shape[0]
    radius = torch.full((n,), float("inf"), device=images.device)
    success = torch.zeros((n,), dtype=torch.bool, device=images.device)

    for epsilon in epsilon_values:
        result = targeted_pgd(
            model,
            images,
            target=target,
            epsilon=epsilon,
            steps=steps,
            alpha=epsilon * float(alpha_fraction),
            random_start=random_start,
            restarts=restarts,
        )
        newly_successful = (~success) & result.success
        radius = torch.where(newly_successful, result.best_linf, radius)
        success = success | result.success

    return GridRadiusResult(
        radius=radius,
        success=success,
        censored=~success,
    )
