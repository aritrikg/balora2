import pytest
import torch
from models.confidence import estimate_uncertainty


def test_rejects_one_sample_and_bad_values():
    with pytest.raises(ValueError, match="two posterior"):
        estimate_uncertainty(torch.zeros(1,1,4,4))
    with pytest.raises(ValueError, match="NaN"):
        estimate_uncertainty(torch.full((2,1,2,2), float("nan")))


def test_uncertainty_shapes_and_bounds():
    logits = torch.stack([torch.full((2,5,4), -2.0), torch.full((2,5,4), 2.0)])
    uq = estimate_uncertainty(logits)
    assert uq.mean_prob.shape == (2,5,4)
    assert uq.epistemic.shape == (2,5,4)
    assert torch.isfinite(uq.epistemic).all()
    assert (uq.epistemic >= 0).all()
    assert float(uq.epistemic.mean()) > 0.2
    same = estimate_uncertainty(torch.full((3,2,5,4), 2.0))
    assert float(same.epistemic.max()) < 1e-6


def test_uncertainty_stays_finite_when_float32_sigmoid_saturates():
    # Large finite logits make float32 sigmoid return exact 0 or 1. The entropy
    # clamp must account for float32 precision so that 0*log(0) never becomes NaN.
    logits = torch.tensor([
        [[[100.0, -100.0, 50.0, -50.0]]],
        [[[101.0, -101.0, 60.0, -60.0]]],
    ], dtype=torch.float32)
    probs = torch.sigmoid(logits)
    assert ((probs == 0.0) | (probs == 1.0)).any()
    uq = estimate_uncertainty(logits)
    for tensor in (uq.mean_prob, uq.total_uncertainty,
                   uq.expected_conditional_entropy, uq.epistemic):
        assert torch.isfinite(tensor).all()
    assert (uq.epistemic >= 0).all()
