"""Lightweight CPU smoke test for BALoRA's math modules (not real SAM2)."""
from __future__ import annotations
import torch
from models.bayesian_lora import BayesianLoRAConfig, BayesianLoRALinear
from models.confidence import estimate_uncertainty
from models.rank_controller import RankController, RankControllerConfig
from losses.losses import dice_bce_loss

def run_smoke_test() -> dict:
    torch.manual_seed(2026)
    layer = BayesianLoRALinear(torch.nn.Linear(16, 8), BayesianLoRAConfig(16, 8, rank=4, alpha=8))
    x = torch.randn(2, 16)
    samples = torch.stack([layer(x) for _ in range(2)])
    assert samples.shape == (2, 2, 8) and torch.isfinite(samples).all()
    assert torch.isfinite(layer.kl_divergence())
    uq = estimate_uncertainty(torch.randn(2, 1, 8, 8))
    assert uq.epistemic.shape == (1, 8, 8) and torch.isfinite(uq.epistemic).all()
    rc = RankController(["decoder.layer"], RankControllerConfig([2, 4, 8]))
    ranks = rc.select_ranks({"decoder.layer": 1.0})
    assert ranks["decoder.layer"] in {2,4,8}
    logits = torch.randn(1,1,8,8,requires_grad=True); targets=(torch.rand(1,1,8,8)>0.5).float()
    loss = dice_bce_loss(logits, targets); loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    return {"bayesian_lora":"PASS","uncertainty":"PASS","rank_controller":"PASS","loss_backward":"PASS",
            "note":"Synthetic CPU smoke test only; this does not exercise official SAM2 or BTCV."}

def main() -> None:
    import json; print(json.dumps(run_smoke_test(), indent=2))

if __name__ == "__main__": main()
