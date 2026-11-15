"""Loss functions (Task 6.6 / Stage 2 eq. 7).

L = L_seg (Dice + BCE) + lambda1 * KL[q_phi || p] + lambda2 * L_cal + lambda3 * L_rank

Not executed in the authoring sandbox — verify with scripts/smoke_test.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Soft Dice loss for binary/multi-channel segmentation.

    Args:
        probs: (B, C, H, W) predicted probabilities in [0, 1] (post-sigmoid
            or post-softmax, caller's choice depending on binary/multiclass).
        targets: (B, C, H, W) one-hot (or binary) ground-truth masks, same
            shape as `probs`.
        eps: smoothing constant to avoid division by zero on empty masks.

    Returns:
        Scalar loss = 1 - mean Dice coefficient across batch and channels.

    Raises:
        ValueError: if `probs` and `targets` shapes don't match.
    """
    if probs.shape != targets.shape:
        raise ValueError(
            f"probs and targets must have the same shape, got {tuple(probs.shape)} "
            f"vs {tuple(targets.shape)}"
        )
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> torch.Tensor:
    """Combined Dice + BCE loss, the standard choice for imbalanced medical
    segmentation masks (BCE gives per-pixel gradient signal even when Dice
    gradients vanish for near-empty masks, and vice versa).

    Args:
        logits: (B, C, H, W) raw (pre-sigmoid) predictions.
        targets: (B, C, H, W) binary ground truth, same shape as `logits`.
        dice_weight, bce_weight: relative weights.

    Returns:
        Scalar combined loss.
    """
    probs = torch.sigmoid(logits)
    d = dice_loss(probs, targets)
    b = F.binary_cross_entropy_with_logits(logits, targets)
    return dice_weight * d + bce_weight * b


def soft_expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """A differentiable (soft-binned) proxy for Expected Calibration Error,
    usable directly as a training loss term (Task 6.6, L_cal).

    Standard ECE bins predictions into hard buckets, which is not
    differentiable. This version uses soft (triangular) bin membership so
    gradients can flow, following the general recipe used in
    calibration-aware training losses (e.g. soft-binned ECE variants in the
    calibration literature cited in the Stage 1 review).

    Args:
        probs: (N,) or (B,H,W) flattened predicted probabilities in [0, 1].
        targets: same shape, binary ground truth in {0, 1}.
        n_bins: number of soft bins across [0, 1].

    Returns:
        Scalar soft-ECE value (lower is better-calibrated).

    Raises:
        ValueError: if shapes don't match.
    """
    if probs.shape != targets.shape:
        raise ValueError("probs and targets must have the same shape")
    p = probs.reshape(-1)
    y = targets.reshape(-1).float()

    if n_bins < 2:
        raise ValueError("n_bins must be >=2")
    # Include both endpoints so p=0 and p=1 receive nonzero membership.
    bin_centers = torch.linspace(0.0, 1.0, n_bins, device=p.device, dtype=p.dtype)
    bin_width = 1.0 / (n_bins - 1)

    # Soft membership: triangular kernel around each bin center.
    # weights: (N, n_bins)
    dist = (p.unsqueeze(1) - bin_centers.unsqueeze(0)).abs()
    weights = torch.clamp(1 - dist / bin_width, min=0.0)
    weights_sum = weights.sum(dim=0).clamp(min=1e-8)

    bin_conf = (weights * p.unsqueeze(1)).sum(dim=0) / weights_sum
    bin_acc = (weights * y.unsqueeze(1)).sum(dim=0) / weights_sum
    bin_mass = weights_sum / weights_sum.sum().clamp(min=1e-8)

    ece = (bin_mass * (bin_conf - bin_acc).abs()).sum()
    return ece


@dataclass
class LossWeights:
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    lambda_kl: float = 1e-4
    lambda_cal: float = 0.1
    lambda_rank: float = 0.01


@dataclass
class LossOutput:
    total: torch.Tensor
    seg: torch.Tensor
    kl: torch.Tensor
    calibration: torch.Tensor
    rank_sparsity: torch.Tensor


def compute_total_loss(
    logits_mean: torch.Tensor,
    targets: torch.Tensor,
    kl_divergence: torch.Tensor,
    rank_sparsity: torch.Tensor,
    weights: LossWeights,
    n_samples_for_kl_scaling: Optional[int] = None,
) -> LossOutput:
    """Combines all loss terms into the total training objective.

    Args:
        logits_mean: (B, C, H, W) mean logits across posterior samples (used
            for both the Dice+BCE term and the calibration term).
        targets: (B, C, H, W) ground truth.
        kl_divergence: scalar KL term, e.g. from
            `models.bayesian_lora.total_kl_divergence`.
        rank_sparsity: scalar rank-usage penalty, from
            `RankController.sparsity_loss()`.
        weights: `LossWeights` dataclass with all lambda coefficients.
        n_samples_for_kl_scaling: if provided, divides the KL term by the
            number of training examples (standard ELBO scaling so the KL
            term doesn't dominate as dataset size grows); if None, KL is
            used as-is and `lambda_kl` alone controls its weight.

    Returns:
        `LossOutput` with the total loss and each component (components are
        useful for logging / diagnosing which term dominates during
        training).
    """
    seg = dice_bce_loss(
        logits_mean, targets, weights.dice_weight, weights.bce_weight
    )

    kl = kl_divergence
    if n_samples_for_kl_scaling is not None and n_samples_for_kl_scaling > 0:
        kl = kl / n_samples_for_kl_scaling

    probs_mean = torch.sigmoid(logits_mean)
    cal = soft_expected_calibration_error(probs_mean, targets)

    total = (
        seg
        + weights.lambda_kl * kl
        + weights.lambda_cal * cal
        + weights.lambda_rank * rank_sparsity
    )

    return LossOutput(
        total=total, seg=seg, kl=kl, calibration=cal, rank_sparsity=rank_sparsity
    )
