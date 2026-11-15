"""Prompt-preserving refinement candidate for a validation-only follow-up.

No target mask or organ identity is accepted here. Lower uncertainty is an
admission rule, not evidence of higher segmentation accuracy. Validate this
candidate before choosing it; keep the original refinement implementation
available to reproduce the already completed experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy import ndimage
import torch

from models.confidence import _binary_entropy, estimate_uncertainty, find_uncertain_regions
from models.refinement import RefinementConfig, _feather_blend, _pad_bbox
from utils.runtime_budget import check_deadline


@dataclass
class GuardedRefinementConfig(RefinementConfig):
    policy: str = "prompt_preserving_guarded_v1"
    min_epistemic_reduction: float = 1e-6
    entropy_tolerance: float = 1e-6


@dataclass
class GuardedTrace:
    iteration: int
    region_bbox: tuple[int, int, int, int]
    prompt_point: tuple[int, int]
    ranks_used: dict[str, int]
    accepted: bool
    reason: str
    epistemic_before: float
    epistemic_after: float | None
    predictive_entropy_before: float
    predictive_entropy_after: float | None


def refine_guarded(
    model: Any,
    image_embedding: Any,
    initial_mean_prob: torch.Tensor,
    initial_epistemic: torch.Tensor,
    original_points_xy: torch.Tensor,
    config: GuardedRefinementConfig,
    deadline: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[GuardedTrace]]:
    """Refine uncertain pixels while retaining the original positive prompt.

    The crop includes the predicted connected component containing the
    original prompt. Proposal attribution and decoding both use that same
    prompt, rather than inventing a positive point at a box center. Only the
    selected uncertainty component may change. Reject a proposal unless its
    epistemic mean decreases and predictive entropy does not increase over
    those exact pixels. A rejected proposal leaves the previous prediction
    intact and ends the loop; uncertainty alone cannot guarantee accuracy.
    """
    if initial_mean_prob.ndim != 2 or initial_epistemic.shape != initial_mean_prob.shape:
        raise ValueError("Expected matching single-image (H,W) probability/uncertainty maps")
    if not torch.isfinite(initial_mean_prob).all() or not torch.isfinite(initial_epistemic).all():
        raise ValueError("Initial probability and uncertainty must be finite")
    if bool(((initial_mean_prob < 0) | (initial_mean_prob > 1)).any()) or bool((initial_epistemic < 0).any()):
        raise ValueError("Initial probabilities must be in [0,1] and epistemic uncertainty non-negative")
    if not 0 < config.confidence_threshold < 1 or config.max_iters < 0:
        raise ValueError("Invalid confidence threshold or iteration count")
    if config.crop_padding < 0 or config.region_min_area < 1 or config.k_samples < 2:
        raise ValueError("Invalid crop padding, component area or sample count")
    if not all(math.isfinite(float(v)) for v in (config.confidence_threshold,
            config.min_epistemic_reduction, config.entropy_tolerance)):
        raise ValueError("Refinement thresholds must be finite")
    if config.min_epistemic_reduction <= 0 or config.entropy_tolerance < 0:
        raise ValueError("Invalid uncertainty acceptance tolerances")
    if config.policy != "prompt_preserving_guarded_v1":
        raise ValueError("Unknown guarded refinement policy")
    if tuple(original_points_xy.shape) != (1, 1, 2) or not torch.isfinite(original_points_xy).all():
        raise ValueError("One finite original positive point is required, shaped (1,1,2)")
    h, w = initial_mean_prob.shape
    px_float, py_float = original_points_xy[0, 0].detach().cpu().tolist()
    if not (0 <= px_float < w and 0 <= py_float < h):
        raise ValueError("Original prompt lies outside the image")
    py, px = int(round(py_float)), int(round(px_float))
    if not (0 <= px < w and 0 <= py < h):
        raise ValueError("Rounded original prompt lies outside the image")
    mean_prob = initial_mean_prob.detach().clone()
    epistemic = initial_epistemic.detach().clone()
    trace: list[GuardedTrace] = []
    point_labels = torch.ones((1, 1), device=mean_prob.device, dtype=torch.int32)
    points = original_points_xy.detach().to(device=mean_prob.device, dtype=torch.float32)
    for iteration in range(config.max_iters):
        check_deadline(deadline, "guarded refinement")
        # Without a predicted target at the known positive point, there is no
        # reliable predicted component to use as crop context. Keep baseline.
        foreground = (mean_prob >= 0.5).cpu().numpy()
        components, _ = ndimage.label(foreground)
        component_id = int(components[py, px])
        if component_id == 0:
            break
        target_component = components == component_id
        regions = find_uncertain_regions(
            epistemic.unsqueeze(0),
            threshold=(1 - config.confidence_threshold) * math.log(2),
            min_area=config.region_min_area,
        )
        region = next((r for r in regions if np.any(r.mask.numpy() & target_component)), None)
        if region is None:
            break
        ys, xs = np.nonzero(target_component | region.mask.numpy())
        bbox = _pad_bbox((int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
                         config.crop_padding, h, w)
        active = region.mask.to(device=mean_prob.device, dtype=torch.bool)
        # The same target prompt is used for attribution and crop decoding.
        ranks, _ = model.select_ranks_from_uncertainty(
            image_embedding, points, point_labels, k_samples=2,
            spatial_mask=active.unsqueeze(0),
        )
        logits_patch = model.decode_region(image_embedding, (py, px), bbox, ranks, config.k_samples)
        before_mi = float(epistemic[active].mean())
        before_entropy = float(_binary_entropy(mean_prob[active], 1e-8).mean())
        if not torch.isfinite(logits_patch).all():
            trace.append(GuardedTrace(iteration, bbox, (py, px), ranks, False,
                                     "nonfinite_proposal", before_mi, None, before_entropy, None))
            del logits_patch
            break
        uq = estimate_uncertainty(logits_patch)
        patch_prob = uq.mean_prob[0].detach()
        patch_epistemic = uq.epistemic[0].detach()
        y0, x0, y1, x1 = bbox
        expected = (y1-y0+1, x1-x0+1)
        if tuple(patch_prob.shape) != expected:
            raise ValueError(f"Regional decoder returned {tuple(patch_prob.shape)}, expected {expected}")
        proposal = _feather_blend(mean_prob, patch_prob, bbox, feather=config.crop_padding//2)
        proposal_mi = _feather_blend(epistemic, patch_epistemic, bbox, feather=config.crop_padding//2)
        # Confident pixels, including pixels inside the rectangle, stay exact.
        proposal = torch.where(active, proposal, mean_prob)
        proposal_mi = torch.where(active, proposal_mi, epistemic)
        after_mi = float(proposal_mi[active].mean())
        after_entropy = float(_binary_entropy(proposal[active], 1e-8).mean())
        if float(proposal[py, px]) < 0.5:
            accepted, reason = False, "positive_prompt_lost"
        elif after_mi > before_mi - config.min_epistemic_reduction:
            accepted, reason = False, "epistemic_not_improved"
        elif after_entropy > before_entropy + config.entropy_tolerance:
            accepted, reason = False, "predictive_entropy_increased"
        else:
            accepted, reason = True, "accepted"
        trace.append(GuardedTrace(iteration, bbox, (py, px), ranks, accepted, reason,
                                 before_mi, after_mi, before_entropy, after_entropy))
        del logits_patch, uq, patch_prob, patch_epistemic
        if not accepted:
            break
        mean_prob, epistemic = proposal.detach(), proposal_mi.detach()
    return mean_prob.detach(), epistemic.detach(), trace
