"""Unmodified, frozen official SAM2 for the zero-shot reference comparison."""
from __future__ import annotations
import torch
from torch import nn
from models.bayesian_lora import BayesianLoRALinear, DeterministicLoRALinear
from models.sam2_wrapper import (Sam2BayesianLoRAModel, SAM2_RGB_MEAN,
                                SAM2_RGB_STD, resolve_sam2_builder)


class FrozenSAM2(nn.Module):
    """Use identical preprocessing and prompt decoding with no adapters.

    The optional backbone argument injects a small CPU test model. Research
    runs load a fresh official backbone, never the trained LoRA checkpoint.
    There is one deterministic prediction, no rank or posterior sampling.
    """
    def __init__(self, cfg: dict, device: str, backbone=None):
        super().__init__()
        self.test_backbone_injected = backbone is not None
        if backbone is None:
            s = cfg['sam2']
            backbone = resolve_sam2_builder()(s['model_cfg'], s['checkpoint'],
                                              device=device, mode='eval')
        if any(isinstance(m, (BayesianLoRALinear, DeterministicLoRALinear))
               for m in backbone.modules()):
            raise ValueError('Zero-shot SAM2 must have no injected LoRA adapters')
        self.backbone = backbone.to(device)
        self.input_size = int(getattr(backbone, 'image_size', 1024))
        self.encoder_amp = bool(cfg['training'].get('amp', False))
        self.register_buffer('_rgb_mean', torch.tensor(SAM2_RGB_MEAN).view(1,3,1,1), persistent=False)
        self.register_buffer('_rgb_std', torch.tensor(SAM2_RGB_STD).view(1,3,1,1), persistent=False)
        self.identity = {
            'method':'official_sam2_zero_shot_v1','model_cfg':cfg['sam2']['model_cfg'],
            'checkpoint_sha256':cfg['sam2'].get('checkpoint_sha256'),
            'resolved_commit':cfg['sam2'].get('resolved_commit'),
            'input_size':self.input_size,'encoder_amp':self.encoder_amp,
            'test_backbone_injected':self.test_backbone_injected,
        }
        self.requires_grad_(False)
        self.eval()

    def encode_image(self, image):
        return Sam2BayesianLoRAModel.encode_image(self, image)

    def decode_one(self, embedding, points, labels):
        return Sam2BayesianLoRAModel._decode_once(self, embedding, points, labels)
