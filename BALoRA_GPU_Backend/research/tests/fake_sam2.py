"""Small differentiable SAM2-API test double. Never used for research metrics."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(3, 3)
        self.conv0 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv1 = nn.Conv2d(4, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)

    def forward(self, x):
        # Exercise an injectable encoder Linear without changing its API.
        x = self.qkv(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f0 = F.avg_pool2d(self.conv0(x), 4)
        f1 = self.conv1(F.avg_pool2d(f0, 2))
        f2 = self.conv2(F.avg_pool2d(f1, 2))
        # SAM2 image encoder returns low-to-high-resolution FPN levels.
        return {"backbone_fpn": [f0, f1, f2],
                "vision_pos_enc": [torch.zeros_like(f0), torch.zeros_like(f1), torch.zeros_like(f2)]}

class FakePromptEncoder(nn.Module):
    def __init__(self, channels=8, grid=4):
        super().__init__(); self.point_proj = nn.Linear(2, channels)
        self.channels, self.grid = channels, grid

    def forward(self, points=None, boxes=None, masks=None):
        coords, labels = points
        scale = coords.new_tensor([64.0, 64.0])
        sparse = self.point_proj(coords / scale) + labels.to(coords.dtype).unsqueeze(-1) * 0.01
        dense = coords.new_zeros(coords.shape[0], self.channels, self.grid, self.grid)
        return sparse, dense

    def get_dense_pe(self):
        return torch.zeros(1, self.channels, self.grid, self.grid)

class FakeMaskDecoder(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.mask_proj = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output=False, repeat_image=False,
                high_res_features=None):
        pooled = image_embeddings.mean(dim=(2, 3)) + sparse_prompt_embeddings.mean(dim=1)
        decoded = torch.tanh(self.out_proj(torch.tanh(self.q_proj(pooled))))
        feature = image_embeddings + decoded[:, :, None, None]
        logits = self.mask_proj(feature)
        # The official mask decoder returns masks, IoU predictions, tokens, object score.
        iou = decoded.mean(-1, keepdim=True)
        tokens = pooled[:, None, :]
        object_score = decoded.mean(-1, keepdim=True)
        return logits, iou, tokens, object_score

class FakeSAM2(nn.Module):
    def __init__(self, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = 8
        self.directly_add_no_mem_embed = False
        self.image_encoder = FakeImageEncoder()
        self.sam_prompt_encoder = FakePromptEncoder(channels=8, grid=image_size // 16)
        self.sam_mask_decoder = FakeMaskDecoder(channels=8)
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 8))

    def forward_image(self, image):
        return self.image_encoder(image)

    def _prepare_backbone_features(self, backbone_out):
        maps = backbone_out["backbone_fpn"]
        pos = backbone_out["vision_pos_enc"]
        sizes = [(x.shape[-2], x.shape[-1]) for x in pos]
        feats = [x.flatten(2).permute(2, 0, 1) for x in maps]
        pos_feats = [x.flatten(2).permute(2, 0, 1) for x in pos]
        return backbone_out, feats, pos_feats, sizes
