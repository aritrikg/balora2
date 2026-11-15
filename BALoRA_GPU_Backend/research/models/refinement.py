"""Confidence-Aware Refinement module (Task 6.5, Stage 2 eq. 6, Algorithm 2).

Given an uncertain region, generates an auxiliary prompt and re-decodes only
that region, then splices the result back into the full mask with a
feathered blend to avoid boundary seams.

This module orchestrates calls into a `Sam2DecoderInterface` (defined in
models/sam2_wrapper.py) rather than implementing SAM2 decoding itself, since
the actual decode call depends on the installed `sam2` package's API.

Not executed in the authoring sandbox — verify with scripts/smoke_test.py
(which uses a mock decoder, not real SAM2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Protocol

import torch
import torch.nn.functional as F
import math

from models.confidence import UncertainRegion, estimate_uncertainty, find_uncertain_regions


class Sam2DecoderInterface(Protocol):
    """Protocol the SAM2 wrapper must satisfy for refinement to call into it.

    Implemented concretely by `models/sam2_wrapper.Sam2BayesianLoRAModel`.
    """

    def decode_region(
        self,
        image_embedding: torch.Tensor,
        prompt_point: tuple[int, int],
        bbox: tuple[int, int, int, int],
        ranks: dict[str, int],
        k_samples: int,
    ) -> torch.Tensor:
        """Returns (K, 1, h, w) logits for the cropped region only."""
        ...


@dataclass
class RefinementConfig:
    confidence_threshold: float = 0.85  # tau
    max_iters: int = 3  # T_max
    crop_padding: int = 16
    region_min_area: int = 25
    k_samples: int = 4


@dataclass
class RefinementTrace:
    """Record of what happened during refinement, for the visualization
    module (Task 6.9) and for debugging."""

    iteration: int
    region_bbox: tuple[int, int, int, int]
    prompt_point: tuple[int, int]
    ranks_used: dict[str, int]
    confidence_before: float
    confidence_after: float


def _auto_prompt_from_region(region: UncertainRegion) -> tuple[int, int]:
    """Picks a prompt point inside an uncertain region.

    Uses the region's bounding-box center as a simple, robust default.
    Stage 2 §6.5 also suggests "point of max uncertainty" or "medial axis"
    as alternatives worth ablating -- flagged here as a clear extension
    point rather than silently picking one without acknowledgment.

    Args:
        region: an `UncertainRegion` from `find_uncertain_regions`.

    Returns:
        (y, x) pixel coordinate to use as the SAM2 point prompt.
    """
    y_min, x_min, y_max, x_max = region.bbox
    return ((y_min + y_max) // 2, (x_min + x_max) // 2)


def _pad_bbox(
    bbox: tuple[int, int, int, int], padding: int, h: int, w: int
) -> tuple[int, int, int, int]:
    """Pads a bbox by `padding` pixels, clamped to image bounds."""
    y_min, x_min, y_max, x_max = bbox
    return (
        max(0, y_min - padding),
        max(0, x_min - padding),
        min(h - 1, y_max + padding),
        min(w - 1, x_max + padding),
    )


def _feather_blend(
    full_mask: torch.Tensor,
    patch: torch.Tensor,
    bbox: tuple[int, int, int, int],
    feather: int = 4,
) -> torch.Tensor:
    """Splices `patch` into `full_mask` at `bbox` with a linear feather at
    the boundary, to avoid a visible seam (Task 6.5 step 5).

    Args:
        full_mask: (H, W) tensor to write into (modified out-of-place;
            returns a new tensor).
        patch: (h, w) tensor matching the bbox region size.
        bbox: (y_min, x_min, y_max, x_max), inclusive, matching patch shape.
        feather: width in pixels of the linear feather ramp at the patch
            boundary.

    Returns:
        New (H, W) tensor with the patch blended in.
    """
    y_min, x_min, y_max, x_max = bbox
    h, w = patch.shape
    result = full_mask.clone()

    # Build a feather weight map: 1.0 in the patch interior, ramping to 0
    # over `feather` pixels at the edges.
    weight = torch.ones_like(patch)
    f = min(feather, h // 2, w // 2)
    if f > 0:
        ramp = torch.linspace(0, 1, f, device=patch.device)
        weight[:f, :] *= ramp.view(-1, 1)
        weight[-f:, :] *= ramp.flip(0).view(-1, 1)
        weight[:, :f] *= ramp.view(1, -1)
        weight[:, -f:] *= ramp.flip(0).view(1, -1)

    region = result[y_min : y_max + 1, x_min : x_max + 1]
    blended = weight * patch + (1 - weight) * region
    result[y_min : y_max + 1, x_min : x_max + 1] = blended
    return result


def refine(
    decoder: Sam2DecoderInterface,
    image_embedding: torch.Tensor,
    initial_mean_prob: torch.Tensor,
    initial_epistemic: torch.Tensor,
    attribution_fn: Callable[[torch.Tensor, UncertainRegion], dict[str, int]],
    config: RefinementConfig,
) -> tuple[torch.Tensor, torch.Tensor, List[RefinementTrace]]:
    """Runs the confidence-aware refinement loop (Algorithm 2, steps 6-14).

    Args:
        decoder: object implementing `Sam2DecoderInterface`, providing
            region-local decoding against a fixed (cached) image embedding.
        image_embedding: cached SAM2 image embedding for the current image.
        initial_mean_prob: (H, W) initial mean predicted probability mask.
        initial_epistemic: (H, W) initial epistemic uncertainty map.
        attribution_fn: callable(epistemic_map, region) -> {layer_name: rank}
            implementing the rank-selection step (eq. 5); typically wraps a
            `RankController.select_ranks` call combined with a gradient-
            based attribution computation upstream.
        config: `RefinementConfig`.

    Returns:
        (refined_mean_prob, refined_epistemic, trace) where trace is a list
        of `RefinementTrace` records (empty if no refinement was needed).

    Raises:
        ValueError: if input tensors are not 2-D (this function operates on
            a single image at a time; batch by calling in a loop or vmap
            upstream).
    """
    if initial_mean_prob.dim() != 2 or initial_epistemic.dim() != 2:
        raise ValueError(
            "refine() operates on a single (H, W) image at a time; got "
            f"shapes {tuple(initial_mean_prob.shape)} and "
            f"{tuple(initial_epistemic.shape)}"
        )

    mean_prob = initial_mean_prob.clone()
    epistemic = initial_epistemic.clone()
    h, w = mean_prob.shape
    trace: List[RefinementTrace] = []

    for iteration in range(config.max_iters):
        if not 0.0 < config.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be in (0,1)")
        if config.max_iters < 0 or config.crop_padding < 0 or config.k_samples < 2:
            raise ValueError("max_iters/crop_padding must be non-negative and k_samples >= 2")
        # Epistemic entropy is bounded by log(2) for binary masks. Convert the
        # requested normalized confidence threshold into the matching entropy.
        regions = find_uncertain_regions(
            epistemic.unsqueeze(0),
            threshold=(1.0 - config.confidence_threshold) * math.log(2.0),
            min_area=config.region_min_area,
        )
        if not regions:
            break  # confident everywhere; done

        region = regions[0]  # worst region first
        confidence_before = max(0.0, min(1.0, 1.0 - region.mean_epistemic / math.log(2.0)))

        prompt_point = _auto_prompt_from_region(region)
        bbox = _pad_bbox(region.bbox, config.crop_padding, h, w)
        ranks = attribution_fn(epistemic, region)

        logits_patch = decoder.decode_region(
            image_embedding=image_embedding,
            prompt_point=prompt_point,
            bbox=bbox,
            ranks=ranks,
            k_samples=config.k_samples,
        )  # (K, 1, ph, pw)

        # estimate_uncertainty expects (K, B, H, W); logits_patch is already
        # (K, 1, ph, pw) with B=1, so it can be passed through directly.
        uq = estimate_uncertainty(logits_patch)

        patch_prob = uq.mean_prob.squeeze(0)  # (ph, pw)
        patch_epistemic = uq.epistemic.squeeze(0)

        mean_prob = _feather_blend(mean_prob, patch_prob, bbox, feather=config.crop_padding // 2)
        epistemic = _feather_blend(epistemic, patch_epistemic, bbox, feather=config.crop_padding // 2)

        confidence_after = max(0.0, min(1.0, 1.0 - float(
            epistemic[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1].mean().item()
        ) / math.log(2.0)))

        trace.append(
            RefinementTrace(
                iteration=iteration,
                region_bbox=bbox,
                prompt_point=prompt_point,
                ranks_used=ranks,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
            )
        )

    return mean_prob, epistemic, trace
