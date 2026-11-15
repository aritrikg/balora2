"""Gradient-attribution-driven discrete adapter-rank controller."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
import torch
import torch.nn as nn

@dataclass
class RankControllerConfig:
    rank_bank: List[int]
    gamma: float = 4.0
    learned_prior_weight: float = 0.2

class RankController(nn.Module):
    """Map each Bayesian decoder layer's attribution to a rank-bank choice.

    The gate is a learned per-layer prior trained by the rank-sparsity term.
    Observed gradient attribution dominates the input-specific decision. With
    all-zero attribution, the lowest rank is selected rather than silently
    consuming the middle of the rank bank.
    """
    def __init__(self, layer_names: List[str], config: RankControllerConfig):
        super().__init__()
        bank = list(config.rank_bank)
        if not bank or any((not isinstance(r, int) or r <= 0) for r in bank):
            raise ValueError("rank_bank must be a non-empty list of positive integers")
        if bank != sorted(set(bank)):
            raise ValueError("rank_bank must be sorted strictly ascending with no duplicates")
        if not layer_names or len(set(layer_names)) != len(layer_names):
            raise ValueError("layer_names must be non-empty and unique")
        if not math.isfinite(config.gamma) or config.gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        if not 0.0 <= config.learned_prior_weight <= 1.0:
            raise ValueError("learned_prior_weight must be in [0,1]")
        self.config = config
        self.layer_names = list(layer_names)
        self.gate = nn.Parameter(torch.zeros(len(layer_names)))
        self._name_to_idx = {name: i for i, name in enumerate(self.layer_names)}

    def rank_usage_scores(self) -> torch.Tensor:
        return torch.sigmoid(self.gate)

    def select_ranks(self, attribution_scores: Dict[str, float]) -> Dict[str, int]:
        unknown = set(attribution_scores) - set(self.layer_names)
        if unknown:
            raise ValueError(f"Unknown attribution layer names: {sorted(unknown)}")
        device, dtype = self.gate.device, self.gate.dtype
        vals = [float(attribution_scores.get(name, 0.0)) for name in self.layer_names]
        if any(not math.isfinite(v) for v in vals):
            raise ValueError("Attribution scores must be finite")
        g = torch.tensor([max(0.0, v) for v in vals], device=device, dtype=dtype)
        max_g = g.max() if g.numel() else g.new_zeros(())
        if max_g.item() <= 1e-12:
            attribution = torch.zeros_like(g)
        else:
            normalized = (g / max_g).clamp(0, 1)
            attribution = torch.sigmoid(self.config.gamma * (normalized - 0.5))
        # Gate's initial score is sigmoid(0)=0.5. Keep it as a modest prior.
        learned = self.rank_usage_scores()
        w = self.config.learned_prior_weight
        score = (1.0 - w) * attribution + w * learned
        # No observed attribution must prefer the smallest rank; the learned
        # prior may only gently lift it when trained away from its initial value.
        if max_g.item() <= 1e-12:
            score = w * learned
        bank = self.config.rank_bank
        out: Dict[str, int] = {}
        for i, name in enumerate(self.layer_names):
            idx = int(torch.round(score[i] * (len(bank) - 1)).clamp(0, len(bank) - 1).item())
            out[name] = bank[idx]
        return out

    def sparsity_loss(self) -> torch.Tensor:
        """Mean learned rank-usage prior, differentiable through the gates."""
        return self.rank_usage_scores().mean()
