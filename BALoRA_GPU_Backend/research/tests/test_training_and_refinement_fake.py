import torch
from torch.utils.data import Dataset, DataLoader
from losses.losses import LossWeights
from training.train import train_one_epoch
from models.refinement import RefinementConfig, refine
from tests.test_sam2_wrapper_fake import build_fake

class OneClip(Dataset):
    def __len__(self): return 1
    def __getitem__(self, idx):
        torch.manual_seed(2027)
        image = torch.rand(1,1,64,64)
        label_int = torch.zeros(1,64,64,dtype=torch.long)
        label_int[:,18:42,20:45] = 1
        labels = torch.nn.functional.one_hot(label_int,num_classes=2).permute(0,3,1,2).float()
        return {"image":image,"label":labels,"case_id":"img0001","slice_indices":[16],"valid_slice_mask":torch.ones(1,dtype=torch.bool)}

def collate(rows):
    return {"image":torch.stack([r["image"] for r in rows]),
            "label":torch.stack([r["label"] for r in rows]),
            "case_ids":[r["case_id"] for r in rows],
            "slice_indices":[r["slice_indices"] for r in rows],
            "valid_slice_mask":torch.stack([r["valid_slice_mask"] for r in rows])}

def test_one_batch_training_updates_parameters():
    torch.manual_seed(5)
    model=build_fake()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    before=model.backbone.sam_mask_decoder.q_proj.mu_B.detach().clone()
    loader=DataLoader(OneClip(),batch_size=1,collate_fn=collate)
    loss=train_one_epoch(model,loader,opt,LossWeights(lambda_kl=1e-5,lambda_cal=0.01,lambda_rank=0.01),
                         k_train=2,device="cpu",grad_clip_norm=1.0,log_every=1,amp=False)
    after=model.backbone.sam_mask_decoder.q_proj.mu_B.detach()
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before,after)


def test_refinement_returns_correct_shapes_and_terminates():
    torch.manual_seed(14)
    model=build_fake(); model.eval()
    image=torch.rand(1,1,64,64)
    emb=model.encode_image(image)
    points=torch.tensor([[[30.,30.]]]); labels=torch.ones(1,1,dtype=torch.int32)
    logits=model.decode(emb,points,labels,k_samples=2)
    uq=__import__("models.confidence",fromlist=["estimate_uncertainty"]).estimate_uncertainty(logits.squeeze(2))
    # Deterministic synthetic uncertainty mask is used only to exercise loop logic.
    initial_epistemic=torch.zeros(64,64)
    initial_epistemic[20:35,20:36]=0.65
    initial_prob=uq.mean_prob[0]
    attr=model.make_region_attribution_fn(emb,labels,k_samples=2)
    refined_prob, refined_uncertainty, trace=refine(
        model,emb,initial_prob,initial_epistemic,attr,
        RefinementConfig(confidence_threshold=0.5,max_iters=1,crop_padding=2,region_min_area=1,k_samples=2))
    assert refined_prob.shape==(64,64) and refined_uncertainty.shape==(64,64)
    assert len(trace)<=1


def test_amp_requested_cpu_falls_back_to_fp32():
    """An AMP request must stay executable on CPU-only component-test environments."""
    torch.manual_seed(19)
    model=build_fake()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    before=model.backbone.sam_mask_decoder.q_proj.mu_B.detach().clone()
    loader=DataLoader(OneClip(),batch_size=1,collate_fn=collate)
    loss=train_one_epoch(model,loader,opt,LossWeights(lambda_kl=1e-5,lambda_cal=0.01,lambda_rank=0.01),
                         k_train=2,device="cpu",grad_clip_norm=1.0,log_every=1,amp=True)
    after=model.backbone.sam_mask_decoder.q_proj.mu_B.detach()
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before,after)
