"""Explicit point-estimate LoRA in both SAM2 encoder and mask decoder."""
from __future__ import annotations
import torch
from torch import nn
from models.bayesian_lora import (BayesianLoRALinear, DeterministicLoRALinear,
                                  inject_deterministic_lora)
from models.sam2_wrapper import (Sam2BayesianLoRAModel, SAM2_RGB_MEAN,
                                SAM2_RGB_STD, resolve_sam2_builder)

METHOD = 'deterministic_fixed_rank_lora_v1'


class DeterministicSAM2(nn.Module):
    """Fresh official backbone, trainable deterministic adapters only.

    No Bayesian model is constructed and no BALoRA weights are loaded. A tiny
    optional backbone is reserved for CPU component tests.
    """
    def __init__(self, cfg, device, backbone=None):
        super().__init__()
        lora = cfg['lora']
        for k in ('encoder_rank', 'decoder_rank'):
            if not isinstance(lora[k], int) or isinstance(lora[k], bool) or lora[k] <= 0:
                raise ValueError('Positive integer fixed ranks are required')
        if lora['alpha'] <= 0 or not 0 <= lora['dropout'] < 1:
            raise ValueError('Invalid LoRA scaling or dropout')
        if not lora['encoder_target_modules'] or not lora['target_modules']:
            raise ValueError('Explicit encoder and decoder targets are required')
        self.test_backbone_injected = backbone is not None
        if backbone is None:
            s=cfg['sam2']
            backbone=resolve_sam2_builder()(s['model_cfg'],s['checkpoint'],device=device,mode='train')
        if any(isinstance(m,(BayesianLoRALinear,DeterministicLoRALinear)) for m in backbone.modules()):
            raise ValueError('Control requires a fresh backbone without existing adapters')
        self.backbone=backbone
        self.backbone.requires_grad_(False)
        self.encoder_lora_layers=inject_deterministic_lora(
            backbone.image_encoder,lora['encoder_target_modules'],rank=lora['encoder_rank'],
            alpha=lora['alpha'],dropout=lora['dropout'])
        self.decoder_lora_layers=inject_deterministic_lora(
            backbone.sam_mask_decoder,lora['target_modules'],rank=lora['decoder_rank'],
            alpha=lora['alpha'],dropout=lora['dropout'])
        self.input_size=int(backbone.image_size)
        self.encoder_amp=bool(cfg['training'].get('amp',False))
        mode=cfg.get('runtime',{}).get('encoder_checkpointing','per_block')
        if mode not in ('off','per_block'):raise ValueError('Invalid encoder checkpointing mode')
        self.encoder_gradient_checkpointing_enabled=(Sam2BayesianLoRAModel._enable_encoder_gradient_checkpointing(self) if mode=='per_block' else False)
        if mode=='off':self.encoder_gradient_checkpointing_mode='off'
        self.register_buffer('_rgb_mean',torch.tensor(SAM2_RGB_MEAN).view(1,3,1,1),persistent=False)
        self.register_buffer('_rgb_std',torch.tensor(SAM2_RGB_STD).view(1,3,1,1),persistent=False)
        self._run_config=cfg
        self.identity={'method':METHOD,'model_cfg':cfg['sam2']['model_cfg'],
            'checkpoint_sha256':cfg['sam2'].get('checkpoint_sha256'),
            'resolved_commit':cfg['sam2'].get('resolved_commit'),
            'input_size':self.input_size,'encoder_amp':self.encoder_amp,
            'encoder_rank':lora['encoder_rank'],'decoder_rank':lora['decoder_rank'],
            'test_backbone_injected':self.test_backbone_injected}
        self.to(device)
        self.assert_control()
        self.eval()

    def assert_control(self):
        if any(isinstance(m,BayesianLoRALinear) for m in self.modules()) or hasattr(self,'rank_controller'):
            raise ValueError('Bayesian or adaptive-rank modules reached the deterministic control')
        trainable=[n for n,p in self.named_parameters() if p.requires_grad]
        if not trainable or any(not n.endswith(('.lora_A','.lora_B')) for n in trainable):
            raise ValueError('Only deterministic LoRA A/B tensors may be trained')
        for name,module in self.backbone.named_modules():
            if isinstance(module,DeterministicLoRALinear):
                required=self.identity['encoder_rank'] if name.startswith('image_encoder.') else self.identity['decoder_rank']
                if module.rank!=required or module._active_rank_override is not None:
                    raise ValueError('All adapters must retain their declared fixed rank')

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module,DeterministicLoRALinear):module.train(mode)
        return self

    def encode_image(self,image):
        return Sam2BayesianLoRAModel.encode_image(self,image)

    def decode_one(self,embedding,points,labels):
        return Sam2BayesianLoRAModel._decode_once(self,embedding,points,labels)
