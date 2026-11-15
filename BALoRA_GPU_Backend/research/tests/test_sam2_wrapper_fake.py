import torch
import pytest
from models.sam2_wrapper import Sam2BayesianLoRAModel, Sam2BayesianLoRAConfig
from models.bayesian_lora import BayesianLoRALinear, DeterministicLoRALinear
from tests.fake_sam2 import FakeSAM2


def build_fake():
    cfg = Sam2BayesianLoRAConfig(
        model_cfg="fake", checkpoint="fake", target_modules=["q_proj", "out_proj"],
        rank_bank=[1,2], alpha=2.0, dropout=0.0, prior_std=1.0,
        init_posterior_std=0.15, device="cpu", encoder_target_modules=["qkv"])
    return Sam2BayesianLoRAModel(cfg, sam2_model=FakeSAM2(image_size=64))


def test_encoder_batch_selection_and_decoder_shapes():
    torch.manual_seed(10)
    model = build_fake()
    x = torch.rand(2,1,64,64)
    emb = model.encode_image(x)
    assert emb["image_embed"].shape == (2,8,4,4)
    assert [tuple(a.shape[-2:]) for a in emb["high_res_feats"]] == [(16,16),(8,8)]
    present = torch.tensor([True, False])
    selected = emb[present]
    assert selected["image_embed"].shape[0] == 1
    assert all(t.shape[0] == 1 for t in selected["high_res_feats"])
    pts = torch.tensor([[[22.0, 31.0]], [[45.0, 16.0]]])
    labels = torch.ones((2,1), dtype=torch.int32)
    logits = model.decode(emb, pts, labels, k_samples=2, output_size=(64,64))
    assert logits.shape == (2,2,1,64,64)
    assert torch.isfinite(logits).all()
    assert all(isinstance(m, BayesianLoRALinear) for m in model.backbone.sam_mask_decoder.modules() if isinstance(m, BayesianLoRALinear))
    assert any(isinstance(m, DeterministicLoRALinear) for m in model.backbone.image_encoder.modules())


def test_adaptive_decode_gradients_and_optimizer_update():
    torch.manual_seed(11)
    model=build_fake(); model.train()
    assert not model.backbone.training
    assert model.backbone.sam_mask_decoder.q_proj.training
    x=torch.rand(2,1,64,64); y=(torch.rand(2,1,64,64)>0.8).float()
    emb=model.encode_image(x); pts=torch.tensor([[[10.,10.]],[[20.,25.]]]); pl=torch.ones(2,1,dtype=torch.int32)
    before=model.backbone.sam_mask_decoder.q_proj.mu_B.detach().clone()
    logits, uq, ranks=model.decode_with_adaptive_rank(emb,pts,pl,k_samples=2)
    assert logits.shape == (2,2,1,64,64)
    assert uq.epistemic.shape == (2,64,64)
    assert ranks and all(v in {1,2} for v in ranks.values())
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.mean(0),y)
    loss.backward()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    opt.step()
    after=model.backbone.sam_mask_decoder.q_proj.mu_B.detach()
    assert not torch.equal(before,after)


def test_rejects_native_image_size_mismatch():
    model=build_fake()
    with pytest.raises(ValueError,match="does not match"):
        model.encode_image(torch.rand(1,1,32,32))


def test_hiera_trunk_checkpoint_wrapper_preserves_gradients():
    model = build_fake()

    class TinyTrunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            h = torch.tanh(self.proj(x))
            # Official Hiera yields multiple stage features; test nested outputs.
            return [h, h.square()]

    trunk = TinyTrunk()
    model.backbone.image_encoder.trunk = trunk
    assert model._enable_encoder_gradient_checkpointing()
    model.train()
    x = torch.randn(3, 4)  # no input gradients, same as a raw CT input
    outputs = trunk(x)
    loss = outputs[0].square().mean() + outputs[1].mean()
    loss.backward()
    assert trunk.calls >= 2, "Checkpointed trunk should be recomputed during backward"
    assert trunk.proj.weight.grad is not None
    assert torch.isfinite(trunk.proj.weight.grad).all()


def test_hiera_blockwise_checkpointing_recomputes_each_block():
    model = build_fake()

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return torch.tanh(self.proj(x))

    class BlockTrunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock(), TinyBlock()])
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    trunk = BlockTrunk()
    model.backbone.image_encoder.trunk = trunk
    assert model._enable_encoder_gradient_checkpointing()
    assert model.encoder_gradient_checkpointing_mode == "per_block"
    assert model.encoder_gradient_checkpointing_block_count == 3
    model.train()
    x = torch.randn(2, 4)
    loss = trunk(x).square().mean()
    loss.backward()
    assert all(block.calls >= 2 for block in trunk.blocks), "Each block should be recomputed during backward"
    assert all(block.proj.weight.grad is not None for block in trunk.blocks)
    assert all(torch.isfinite(block.proj.weight.grad).all() for block in trunk.blocks)
