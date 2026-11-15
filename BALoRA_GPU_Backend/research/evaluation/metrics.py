"""Evaluation metric suite (Task 10).

Implements: Dice, IoU, Precision, Recall, F1, HD95, ASSD, ECE, Brier score,
NLL. ROC/PR curves and reliability diagrams are left to the visualization
module / sklearn (see evaluate.py) since sklearn already implements those
robustly -- no need to reimplement.

Not executed in the authoring sandbox — verify with scripts/smoke_test.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import torch

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial.distance import directed_hausdorff
except ImportError:  # pragma: no cover
    distance_transform_edt = None
    directed_hausdorff = None


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Dice similarity coefficient for a single binary mask pair.

    Args:
        pred: (H, W) binary (0/1) predicted mask.
        target: (H, W) binary ground-truth mask.
        eps: smoothing constant.

    Returns:
        Dice score in [0, 1]. Returns 1.0 if both masks are empty (edge
        case: correctly predicting "nothing here").
    """
    pred_f = pred.float().flatten()
    target_f = target.float().flatten()
    intersection = (pred_f * target_f).sum()
    denom = pred_f.sum() + target_f.sum()
    if denom.item() == 0:
        return 1.0
    return float((2 * intersection + eps) / (denom + eps))


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Intersection-over-Union for a single binary mask pair."""
    pred_f = pred.bool().flatten()
    target_f = target.bool().flatten()
    intersection = (pred_f & target_f).sum().float()
    union = (pred_f | target_f).sum().float()
    if union.item() == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def precision_recall_f1(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> Dict[str, float]:
    """Pixel-wise precision, recall, F1 for a single binary mask pair."""
    pred_f = pred.bool().flatten()
    target_f = target.bool().flatten()
    tp = (pred_f & target_f).sum().float()
    fp = (pred_f & ~target_f).sum().float()
    fn = (~pred_f & target_f).sum().float()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Extracts boundary pixel coordinates of a binary mask via erosion."""
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    return np.argwhere(boundary)


def hausdorff_distance_95(pred: torch.Tensor, target: torch.Tensor) -> float:
    """95th-percentile Hausdorff Distance (HD95) between mask boundaries.

    Unlike the standard (max) Hausdorff distance, HD95 is robust to a
    handful of outlier boundary pixels, which is why it's the standard
    choice in medical segmentation benchmarks over plain HD.

    Args:
        pred: (H, W) binary predicted mask.
        target: (H, W) binary ground-truth mask.

    Returns:
        HD95 in pixels. Returns 0.0 if both masks are empty, and
        float('inf') if exactly one is empty (undefined boundary distance).

    Raises:
        RuntimeError: if scipy is not installed.
    """
    if distance_transform_edt is None:
        raise RuntimeError("scipy is required for hausdorff_distance_95.")

    pred_np = pred.bool().cpu().numpy()
    target_np = target.bool().cpu().numpy()

    if not pred_np.any() and not target_np.any():
        return 0.0
    if not pred_np.any() or not target_np.any():
        return float("inf")

    pred_pts = _surface_points(pred_np)
    target_pts = _surface_points(target_np)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("inf")

    # Distance from every pred boundary point to nearest target boundary
    # point, and vice versa, via a distance-transform lookup (fast, avoids
    # an O(N*M) brute-force pairwise distance computation).
    from scipy.ndimage import binary_erosion
    target_dt = distance_transform_edt(~(target_np & ~binary_erosion(target_np)))
    pred_dt = distance_transform_edt(~(pred_np & ~binary_erosion(pred_np)))

    d_pred_to_target = target_dt[pred_pts[:, 0], pred_pts[:, 1]]
    d_target_to_pred = pred_dt[target_pts[:, 0], target_pts[:, 1]]

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_distances, 95))


def average_symmetric_surface_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Average Symmetric Surface Distance (ASSD) between mask boundaries.

    Args:
        pred, target: (H, W) binary masks.

    Returns:
        ASSD in pixels (mean, not 95th percentile, unlike HD95).
    """
    if distance_transform_edt is None:
        raise RuntimeError("scipy is required for average_symmetric_surface_distance.")

    pred_np = pred.bool().cpu().numpy()
    target_np = target.bool().cpu().numpy()

    if not pred_np.any() and not target_np.any():
        return 0.0
    if not pred_np.any() or not target_np.any():
        return float("inf")

    pred_pts = _surface_points(pred_np)
    target_pts = _surface_points(target_np)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("inf")

    from scipy.ndimage import binary_erosion
    target_dt = distance_transform_edt(~(target_np & ~binary_erosion(target_np)))
    pred_dt = distance_transform_edt(~(pred_np & ~binary_erosion(pred_np)))

    d_pred_to_target = target_dt[pred_pts[:, 0], pred_pts[:, 1]]
    d_target_to_pred = pred_dt[target_pts[:, 0], target_pts[:, 1]]

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(all_distances.mean())


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """Standard (hard-binned) Expected Calibration Error, for reporting
    (as opposed to `losses.losses.soft_expected_calibration_error`, which
    is the differentiable training-time version).

    Args:
        probs: flattened predicted probabilities in [0, 1].
        targets: flattened binary ground truth.
        n_bins: number of equal-width confidence bins.

    Returns:
        ECE scalar (lower is better).
    """
    probs_np = probs.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs_np)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs_np > lo) & (probs_np <= hi) if i > 0 else (probs_np >= lo) & (probs_np <= hi)
        if in_bin.sum() == 0:
            continue
        bin_conf = probs_np[in_bin].mean()
        bin_acc = targets_np[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Brier score: mean squared error between predicted probability and
    the binary outcome. Lower is better (0 = perfect)."""
    return float(torch.mean((probs.flatten() - targets.flatten().float()) ** 2))


def negative_log_likelihood(
    probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> float:
    """Mean binary NLL: -[y*log(p) + (1-y)*log(1-p)]."""
    p = probs.float().flatten()
    eps = max(float(eps), float(torch.finfo(p.dtype).eps))
    p = p.clamp(min=eps, max=1 - eps)
    y = targets.flatten().float()
    nll = -(y * torch.log(p) + (1 - y) * torch.log1p(-p))
    return float(nll.mean())


@dataclass
class SegmentationMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    hd95: float
    assd: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CalibrationMetrics:
    ece: float
    brier: float
    nll: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_segmentation_metrics(
    pred: torch.Tensor, target: torch.Tensor, compute_surface_distances: bool = True
) -> SegmentationMetrics:
    """Convenience wrapper computing the full segmentation metric suite for
    one (H, W) binary mask pair.

    Args:
        pred: (H, W) binary predicted mask.
        target: (H, W) binary ground-truth mask.
        compute_surface_distances: if False, skips HD95/ASSD (they're the
            most expensive metrics here) and returns NaN for them --
            useful for a fast per-batch training-time sanity check where
            you only want Dice/IoU, saving surface-distance computation for
            full validation/test runs.

    Returns:
        `SegmentationMetrics` dataclass.
    """
    pr = precision_recall_f1(pred, target)
    hd95 = hausdorff_distance_95(pred, target) if compute_surface_distances else float("nan")
    assd = (
        average_symmetric_surface_distance(pred, target)
        if compute_surface_distances
        else float("nan")
    )
    return SegmentationMetrics(
        dice=dice_coefficient(pred, target),
        iou=iou_score(pred, target),
        precision=pr["precision"],
        recall=pr["recall"],
        f1=pr["f1"],
        hd95=hd95,
        assd=assd,
    )


def compute_calibration_metrics(
    probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15
) -> CalibrationMetrics:
    """Convenience wrapper computing ECE, Brier, NLL for one prediction."""
    return CalibrationMetrics(
        ece=expected_calibration_error(probs, target, n_bins=n_bins),
        brier=brier_score(probs, target),
        nll=negative_log_likelihood(probs, target),
    )
