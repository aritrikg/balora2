import torch
from models.bayesian_lora import BayesianLoRAConfig, BayesianLoRALinear, DeterministicLoRALinear, inject_bayesian_lora


def test_bayesian_lora_shapes_sampling_kl_gradients():
    torch.manual_seed(17)
    layer = BayesianLoRALinear(torch.nn.Linear(5, 4), BayesianLoRAConfig(5,4,rank=2,alpha=4))
    x = torch.randn(3,5)
    y1, y2 = layer(x), layer(x)
    assert y1.shape == (3,4) and torch.isfinite(y1).all()
    assert not torch.equal(y1, y2), "posterior draws should vary"
    kl = layer.kl_divergence()
    assert kl.ndim == 0 and torch.isfinite(kl) and kl.item() >= 0
    (y1.square().mean() + kl * 1e-5).backward()
    assert layer.mu_A.grad is not None and torch.isfinite(layer.mu_A.grad).all()
    assert layer.log_sigma_A.grad is not None


def test_bayesian_lora_deterministic_mode():
    layer = BayesianLoRALinear(torch.nn.Linear(5,4), BayesianLoRAConfig(5,4,rank=2))
    layer.sample_posterior = False
    x = torch.randn(2,5)
    assert torch.equal(layer(x), layer(x))


def test_injection_wraps_only_matching_linears():
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.ReLU(), torch.nn.Linear(4,2))
    names = inject_bayesian_lora(model, ["0"], rank=2)
    assert names == ["0"]
    assert isinstance(model[0], BayesianLoRALinear)
    assert not any(p.requires_grad for p in model[0].base_layer.parameters())


def test_deterministic_lora_does_not_retain_large_forward_activation():
    layer = DeterministicLoRALinear(torch.nn.Linear(8, 6), rank=3, alpha=6.0)
    x = torch.randn(2, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 6)
    assert torch.isfinite(y).all()
    assert layer.last_adapter_output is None
    y.square().mean().backward()
    assert layer.lora_A.grad is not None and torch.isfinite(layer.lora_A.grad).all()
    assert layer.lora_B.grad is not None and torch.isfinite(layer.lora_B.grad).all()
