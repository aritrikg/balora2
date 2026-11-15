"""Prompt sampling utilities.

Fixes Critical Issue #3 from the implementation readiness review: training
and evaluation previously used a single fixed center-image point prompt and
collapsed all organ classes into one "any organ" binary target. This module
provides real per-class foreground point sampling from ground-truth masks,
which `training/train.py` and `evaluation/evaluate.py` now use to prompt
and supervise SAM2 per organ class, not as one undifferentiated blob.
"""
from __future__ import annotations

from typing import Optional

import torch


def sample_foreground_points(mask: torch.Tensor) -> Optional[torch.Tensor]:
    """Samples one random foreground pixel per image in a batch of masks.

    Args:
        mask: (N, H, W) binary mask batch. Every image in the batch MUST
            have at least one foreground pixel -- filter to present-only
            images (e.g. via `mask.flatten(1).any(dim=1)`) before calling
            this, since a fully-empty mask has no valid point to sample.

    Returns:
        (N, 1, 2) tensor of (x, y) pixel coordinates, one per image, or
        None if `mask` has zero images (N == 0) -- callers should check for
        this and skip the class entirely for this batch.

    Raises:
        ValueError: if any image in the batch has no foreground pixels at
            all (this is a caller error -- filter first).
    """
    if mask.shape[0] == 0:
        return None

    n, h, w = mask.shape
    points = torch.zeros(n, 1, 2, dtype=torch.float32, device=mask.device)
    for i in range(n):
        ys, xs = torch.nonzero(mask[i], as_tuple=True)
        if ys.numel() == 0:
            raise ValueError(
                f"sample_foreground_points: image index {i} has no "
                "foreground pixels. Filter out empty-mask images before "
                "calling this function (e.g. via "
                "`present = mask.flatten(1).any(dim=1)`)."
            )
        idx = torch.randint(0, ys.numel(), (1,), device=mask.device).item()
        points[i, 0, 0] = xs[idx].float()
        points[i, 0, 1] = ys[idx].float()
    return points