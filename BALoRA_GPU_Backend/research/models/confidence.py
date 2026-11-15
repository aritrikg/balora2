"""Monte-Carlo posterior predictive uncertainty for binary organ masks.

Input logits have shape (K,B,H,W). The expected per-sample Bernoulli entropy
is reported as `expected_conditional_entropy`; it is not claimed to be a
separately learned aleatoric variance. Epistemic uncertainty is the mutual
information between the prediction and posterior weights, approximated by
predictive entropy minus expected conditional entropy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
try:
    from scipy import ndimage
except ImportError:  # connected-component refinement only
    ndimage = None

@dataclass
class UncertaintyMaps:
    mean_prob: torch.Tensor
    total_uncertainty: torch.Tensor
    expected_conditional_entropy: torch.Tensor
    epistemic: torch.Tensor
    @property
    def aleatoric(self) -> torch.Tensor:
        """Compatibility alias; interpret as expected conditional entropy, not a separately learned component."""
        return self.expected_conditional_entropy


def _binary_entropy(p: torch.Tensor, eps: float) -> torch.Tensor:
    """Numerically stable Bernoulli entropy, including saturated float32 probabilities.

    The default eps=1e-8 is smaller than float32 spacing near 1.0. In float32,
    ``1.0 - 1e-8`` rounds to exactly 1.0, so the old clamp could still produce
    log(0) and the indeterminate expression 0 * -inf when sigmoid saturated.
    Clamp by at least the dtype's machine epsilon.
    """
    if not p.is_floating_point():
        raise TypeError(f"Entropy probabilities must be floating point, got {p.dtype}")
    if not (0.0 < float(eps) < 0.5):
        raise ValueError(f"eps must be between 0 and 0.5, got {eps}")
    effective_eps = max(float(eps), float(torch.finfo(p.dtype).eps))
    p = p.clamp(min=effective_eps, max=1.0-effective_eps)
    return -(p * torch.log(p) + (1-p) * torch.log1p(-p))


def estimate_uncertainty(logits_samples: torch.Tensor, eps: float = 1e-8) -> UncertaintyMaps:
    if logits_samples.ndim != 4:
        raise ValueError(f"Expected (K,B,H,W), got {tuple(logits_samples.shape)}")
    if logits_samples.shape[0] < 2:
        raise ValueError(f"At least two posterior samples are required; got K={logits_samples.shape[0]}")
    if not torch.isfinite(logits_samples).all():
        raise ValueError("logits_samples contains NaN or infinity")
    probs = torch.sigmoid(logits_samples.float())
    mean_prob = probs.mean(0)
    predictive_entropy = _binary_entropy(mean_prob, eps)
    conditional_entropy = _binary_entropy(probs, eps).mean(0)
    epistemic = (predictive_entropy - conditional_entropy).clamp_min(0.0)
    return UncertaintyMaps(mean_prob, predictive_entropy, conditional_entropy, epistemic)

@dataclass
class UncertainRegion:
    batch_index: int
    mask: torch.Tensor
    bbox: tuple[int, int, int, int]
    mean_epistemic: float
    area: int


def find_uncertain_regions(epistemic: torch.Tensor, threshold: float, min_area: int = 25) -> List[UncertainRegion]:
    if epistemic.ndim != 3:
        raise ValueError(f"Expected epistemic shape (B,H,W), got {tuple(epistemic.shape)}")
    if min_area < 1:
        raise ValueError("min_area must be >=1")
    if ndimage is None:
        raise RuntimeError("scipy is required for connected-component refinement; install scipy")
    arr = epistemic.detach().float().cpu().numpy()
    out: List[UncertainRegion] = []
    for b in range(arr.shape[0]):
        labels, n = ndimage.label(arr[b] > float(threshold))
        for idx in range(1, n + 1):
            comp = labels == idx
            area = int(comp.sum())
            if area < min_area:
                continue
            ys, xs = comp.nonzero()
            out.append(UncertainRegion(
                int(b), torch.from_numpy(comp),
                (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
                float(arr[b][comp].mean()), area))
    return sorted(out, key=lambda r: r.mean_epistemic, reverse=True)


def region_confidence(regions: List[UncertainRegion]) -> List[float]:
    # Binary entropy's theoretical upper bound is log(2); normalize before
    # converting an entropy-like epistemic quantity to a [0,1] confidence.
    import math
    return [max(0.0, min(1.0, 1.0-r.mean_epistemic/math.log(2.0))) for r in regions]
