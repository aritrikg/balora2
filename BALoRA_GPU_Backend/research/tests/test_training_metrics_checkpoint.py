import torch
from pathlib import Path
from losses.losses import soft_expected_calibration_error, dice_bce_loss
from evaluation.metrics import compute_segmentation_metrics, compute_calibration_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint


def test_soft_calibration_loss_covers_probability_endpoints():
    probs=torch.zeros(1,1,4,4,requires_grad=True)
    target=torch.ones_like(probs)
    loss=soft_expected_calibration_error(probs,target,n_bins=5)
    assert torch.isfinite(loss) and float(loss.detach())>0
    loss.backward(); assert probs.grad is not None and torch.isfinite(probs.grad).all()


def test_loss_and_metrics_are_well_formed():
    logits=torch.randn(2,1,8,8,requires_grad=True)
    target=(torch.rand(2,1,8,8)>0.7).float()
    loss=dice_bce_loss(logits,target); loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    pred=target[0,0].bool()
    seg=compute_segmentation_metrics(pred,target[0,0].bool(),compute_surface_distances=False)
    assert seg.dice == 1.0 and seg.iou == 1.0
    cal=compute_calibration_metrics(torch.full_like(target[0,0],0.5),target[0,0])
    assert 0 <= cal.ece <= 1 and 0 <= cal.brier <= 1


def test_checkpoint_roundtrip_and_resume_epoch(tmp_path):
    m=torch.nn.Linear(3,2); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5)
    x=torch.ones(1,3); m(x).sum().backward(); opt.step(); sched.step()
    expected={k:v.detach().clone() for k,v in m.state_dict().items()}
    cp=tmp_path/"resume.pt"
    save_checkpoint(m,opt,epoch=3,path=str(cp),scheduler=sched,extra={"tag":"test"})
    m2=torch.nn.Linear(3,2); opt2=torch.optim.AdamW(m2.parameters(),lr=1e-3)
    sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=5)
    start=load_checkpoint(m2,opt2,str(cp),scheduler=sched2,restore_rng=False)
    assert start == 4
    assert all(torch.equal(expected[k],m2.state_dict()[k]) for k in expected)
    assert sched.state_dict() == sched2.state_dict()
