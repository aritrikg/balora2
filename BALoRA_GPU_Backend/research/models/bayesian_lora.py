"""Point-estimate and variational Bayesian LoRA modules used by BALoRA.

The research design uses deterministic LoRA adapters in the image encoder and
Bayesian LoRA adapters in the SAM2 mask decoder. The frozen pretrained Linear
weights are preserved inside each wrapper. This module has CPU-testable
independent components; SAM2 integration tests require the official SAM2 model.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn


@dataclass
class BayesianLoRAConfig:
    in_features: int
    out_features: int
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    prior_std: float = 1.0
    init_posterior_std: float = 0.05

    def validate(self) -> None:
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if self.rank <= 0 or self.rank > min(self.in_features, self.out_features):
            raise ValueError("rank must be positive and no larger than either layer dimension")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if self.prior_std <= 0 or self.init_posterior_std <= 0:
            raise ValueError("prior_std and init_posterior_std must be positive")


class DeterministicLoRALinear(nn.Module):
    """Standard point-estimate LoRA wrapper for encoder-side adaptation."""
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base_layer).__name__}")
        if not 0 < rank <= min(base_layer.in_features, base_layer.out_features):
            raise ValueError(f"Invalid rank {rank} for Linear({base_layer.in_features},{base_layer.out_features})")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.last_adapter_output: Optional[torch.Tensor] = None
        self._active_rank_override: Optional[int] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        r = self._active_rank_override or self.rank
        if not 0 < r <= self.rank:
            raise ValueError(f"Active rank {r} is invalid for configured rank {self.rank}")
        update = self.dropout(x) @ self.lora_A[:r].T @ self.lora_B[:, :r].T
        # Encoder LoRA outputs are not used for attribution. Retaining every
        # high-resolution update on its module pins large activation tensors
        # throughout the entire SAM2 encoder forward and can exhaust a T4 GPU.
        # Keep the compatibility attribute empty; the Bayesian decoder wrapper
        # still retains last_adapter_output because adaptive attribution uses it.
        self.last_adapter_output = None
        return base + (self.alpha / r) * update


class BayesianLoRALinear(nn.Module):
    """Variational LoRA wrapper with factorized Gaussian posterior over A and B.

    The approximate posterior is q(A,B)=q(A)q(B), with diagonal Gaussian
    factors and a zero-mean isotropic Gaussian prior. Sampling uses the
    reparameterization trick. `sample_posterior` is controlled by the wrapper
    for each decoder call, rather than relying on train/eval mode alone.
    """
    def __init__(self, base_layer: nn.Linear, config: BayesianLoRAConfig) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base_layer).__name__}")
        config.validate()
        if (base_layer.in_features, base_layer.out_features) != (config.in_features, config.out_features):
            raise ValueError("LoRA dimensions do not match the wrapped Linear layer")
        self.config = config
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        r, din, dout = config.rank, config.in_features, config.out_features
        self.mu_A = nn.Parameter(torch.empty(r, din))
        self.mu_B = nn.Parameter(torch.zeros(dout, r))
        nn.init.kaiming_uniform_(self.mu_A, a=math.sqrt(5))
        init_log_sigma = math.log(config.init_posterior_std)
        self.log_sigma_A = nn.Parameter(torch.full((r, din), init_log_sigma))
        self.log_sigma_B = nn.Parameter(torch.full((dout, r), init_log_sigma))
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        self.scaling = float(config.alpha) / r
        self.sample_posterior = True
        self._active_rank_override: Optional[int] = None
        self.last_adapter_output: Optional[torch.Tensor] = None

    def sample_weights(self, sample: bool = True,
                       generator: Optional[torch.Generator] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not sample:
            return self.mu_A, self.mu_B
        sigma_A = torch.exp(self.log_sigma_A.clamp(-12.0, 6.0))
        sigma_B = torch.exp(self.log_sigma_B.clamp(-12.0, 6.0))
        eps_A = torch.randn(self.mu_A.shape, dtype=self.mu_A.dtype, device=self.mu_A.device, generator=generator)
        eps_B = torch.randn(self.mu_B.shape, dtype=self.mu_B.dtype, device=self.mu_B.device, generator=generator)
        return self.mu_A + sigma_A * eps_A, self.mu_B + sigma_B * eps_B

    def forward(self, x: torch.Tensor, sample: Optional[bool] = None) -> torch.Tensor:
        """Forward pass, optionally overriding posterior sampling for this call."""
        base = self.base_layer(x)
        A, B = self.sample_weights(sample=self.sample_posterior if sample is None else bool(sample))
        r = self._active_rank_override or self.config.rank
        if not 0 < r <= self.config.rank:
            raise ValueError(f"Active rank {r} is invalid for configured rank {self.config.rank}")
        update = self.dropout(x) @ A[:r].T @ B[:, :r].T
        self.last_adapter_output = update
        return base + (self.config.alpha / r) * update

    def kl_divergence(self) -> torch.Tensor:
        prior_std = float(self.config.prior_std)
        prior_var = prior_std ** 2
        def kl_one(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
            ls = log_sigma.clamp(-12.0, 6.0)
            sigma2 = torch.exp(2 * ls)
            return (math.log(prior_std) - ls + (sigma2 + mu.square()) / (2 * prior_var) - 0.5).sum()
        return kl_one(self.mu_A, self.log_sigma_A) + kl_one(self.mu_B, self.log_sigma_B)

    def effective_rank_subset(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 < rank <= self.config.rank:
            raise ValueError(f"rank must be in (0, {self.config.rank}], got {rank}")
        return self.mu_A[:rank], self.mu_B[:, :rank]


def _inject(module: nn.Module, target_module_names: list[str], wrapper_kind: str,
            rank: int, alpha: float, dropout: float, prior_std: float = 1.0,
            init_posterior_std: float = 0.05) -> list[str]:
    if not target_module_names:
        raise ValueError("target_module_names must not be empty")
    # Snapshot before replacing children: avoids traversing newly inserted wrappers.
    candidates = list(module.named_modules())
    targets = [(name, child) for name, child in candidates
               if name and isinstance(child, nn.Linear)
               and any(token in name for token in target_module_names)]
    wrapped: list[str] = []
    for name, child in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        if wrapper_kind == "bayesian":
            cfg = BayesianLoRAConfig(child.in_features, child.out_features, rank, alpha,
                                     dropout, prior_std, init_posterior_std)
            replacement: nn.Module = BayesianLoRALinear(child, cfg)
        else:
            replacement = DeterministicLoRALinear(child, rank, alpha, dropout)
        setattr(parent, attr, replacement)
        wrapped.append(name)
    if not wrapped:
        candidates_text = [name for name, child in candidates if name and isinstance(child, nn.Linear)]
        raise RuntimeError(
            f"No nn.Linear layers matched {target_module_names}. Inspect actual SAM2 module names. "
            f"First linear module names: {candidates_text[:40]}"
        )
    return wrapped


def inject_bayesian_lora(module: nn.Module, target_module_names: list[str], rank: int = 8,
                         alpha: float = 16.0, dropout: float = 0.0, prior_std: float = 1.0,
                         init_posterior_std: float = 0.05) -> list[str]:
    """Replace matching Linear submodules with Bayesian LoRA modules."""
    return _inject(module, target_module_names, "bayesian", rank, alpha, dropout,
                   prior_std, init_posterior_std)


def inject_deterministic_lora(module: nn.Module, target_module_names: list[str], rank: int = 8,
                              alpha: float = 16.0, dropout: float = 0.0) -> list[str]:
    """Replace matching Linear submodules with point-estimate LoRA modules."""
    return _inject(module, target_module_names, "deterministic", rank, alpha, dropout)


def total_kl_divergence(module: nn.Module) -> torch.Tensor:
    values = [m.kl_divergence() for m in module.modules() if isinstance(m, BayesianLoRALinear)]
    if values:
        return torch.stack(values).sum()
    # Return a scalar on the module's parameter device where possible.
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.new_zeros(())
    return torch.zeros(())


@contextmanager
def sampling_mode(module: nn.Module, sample: bool, ranks: Optional[dict[str, int]] = None) -> Iterator[None]:
    """Temporarily set Bayesian sampling and rank overrides; always restore state."""
    old: list[tuple[nn.Module, bool, Optional[int]]] = []
    try:
        for name, child in module.named_modules():
            if isinstance(child, BayesianLoRALinear):
                old.append((child, child.sample_posterior, child._active_rank_override))
                child.sample_posterior = bool(sample)
                child._active_rank_override = ranks.get(name) if ranks else None
        yield
    finally:
        for child, old_sample, old_rank in old:
            child.sample_posterior = old_sample
            child._active_rank_override = old_rank
