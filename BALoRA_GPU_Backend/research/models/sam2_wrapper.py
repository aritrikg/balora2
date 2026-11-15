"""Verified-shape SAM2 adapter for BALoRA.

This module targets the official SAM2 image API (SAM2Base.forward_image,
_prepare_backbone_features, sam_prompt_encoder and sam_mask_decoder). No
feature maps are resized to disguise channel/spatial incompatibilities. The
model's configured `image_size` is authoritative and the BTCV dataset must use
that same input size. Hardware-dependent integration tests still need the
real checkpoint in Kaggle.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional
import importlib
import weakref
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import torch.nn.functional as F

from models.bayesian_lora import (BayesianLoRALinear, inject_bayesian_lora,
    inject_deterministic_lora, total_kl_divergence, sampling_mode)
from models.confidence import estimate_uncertainty, UncertaintyMaps
from models.rank_controller import RankController, RankControllerConfig

# Do not import the optional official package while BALoRA metadata/helpers
# are imported. A notebook can install SAM2 later in this same interpreter.
build_sam2 = None
_SAM2_IMPORT_ERROR: Optional[Exception] = None


def resolve_sam2_builder():
    """Resolve the currently available official builder at model construction.

    An earlier missing package must not become a permanent cached failure.
    Resolve again even after a previous success, so a newly selected source
    checkout cannot leave a stale builder reference in this module.
    """
    global build_sam2, _SAM2_IMPORT_ERROR
    importlib.invalidate_caches()
    try:
        builder = importlib.import_module("sam2.build_sam").build_sam2
        if not callable(builder):
            raise TypeError("sam2.build_sam.build_sam2 is not callable")
    except Exception as exc:
        build_sam2 = None
        _SAM2_IMPORT_ERROR = exc
        raise ImportError(
            "Could not import the official SAM2 builder at model construction. "
            "Run the pinned SAM2 setup cell, verify its source path and helper dependencies, "
            "then rerun this cell. "
            f"Current import error: {type(exc).__name__}: {exc}"
        ) from exc
    build_sam2 = builder
    _SAM2_IMPORT_ERROR = None
    return builder

# Official SAM2Transforms defaults: input images are converted to [0,1] and
# normalized with these values after resize. The BALoRA dataset maps CT HU to
# [0,1], replicates to RGB and applies the same normalization.
SAM2_RGB_MEAN = (0.485, 0.456, 0.406)
SAM2_RGB_STD = (0.229, 0.224, 0.225)

@dataclass
class Sam2BayesianLoRAConfig:
    model_cfg: str
    checkpoint: str
    target_modules: list[str]
    rank_bank: list[int]
    alpha: float = 16.0
    dropout: float = 0.0
    prior_std: float = 1.0
    init_posterior_std: float = 0.05
    device: str = "cuda"
    encoder_target_modules: list[str] | None = None
    point_estimate_module_substrings: list[str] | None = None
    attribution_gamma: float = 4.0
    learned_prior_weight: float = 0.2

    def validate(self) -> None:
        if not self.rank_bank or self.rank_bank != sorted(set(self.rank_bank)):
            raise ValueError("rank_bank must be sorted ascending with unique positive ranks")
        if any(not isinstance(r, int) or r <= 0 for r in self.rank_bank):
            raise ValueError("rank_bank values must be positive integers")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.prior_std <= 0 or self.init_posterior_std <= 0:
            raise ValueError("Bayesian standard deviations must be positive")
        if not self.target_modules:
            raise ValueError("target_modules for decoder LoRA must not be empty")

class SAM2EmbeddingBundle(dict):
    """Batch-indexable bundle compatible with `embedding[boolean_mask]`.

    Batch-dependent tensors are selected consistently, including all high-res
    feature maps and the retained raw CT tensor. Non-batch metadata is kept.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_size = int(self.get("image_embed").shape[0])

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return super().__getitem__(key)
        if isinstance(key, int):
            key = slice(key, key + 1) if key >= 0 else slice(key, None)
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool and key.ndim != 1:
            raise IndexError(f"Boolean embedding index must be 1-D, got {tuple(key.shape)}")
        if isinstance(key, torch.Tensor) and key.ndim == 1 and key.numel() != self._batch_size:
            raise IndexError(f"Embedding index length {key.numel()} != batch size {self._batch_size}")
        selected: dict[str, Any] = {}
        for name, value in self.items():
            if torch.is_tensor(value) and value.ndim and value.shape[0] == self._batch_size:
                selected[name] = value[key]
            elif isinstance(value, (list, tuple)) and value and all(torch.is_tensor(v) and v.ndim and v.shape[0] == self._batch_size for v in value):
                selected[name] = type(value)(v[key] for v in value)
            else:
                selected[name] = value
        return SAM2EmbeddingBundle(selected)

    @property
    def batch_size(self) -> int:
        return self._batch_size


class Sam2BayesianLoRAModel(nn.Module):
    """SAM2 with deterministic encoder LoRA and Bayesian decoder LoRA.

    `sam2_model` is only intended for tests injecting a small fake model that
    follows the official SAM2 API. For research, omit it so the official
    `build_sam2` path loads the requested checkpoint.
    """
    def __init__(self, config: Sam2BayesianLoRAConfig, sam2_model: nn.Module | None = None) -> None:
        super().__init__(); config.validate(); self.config = config
        if sam2_model is None:
            builder = resolve_sam2_builder()
            try:
                sam2_model = builder(config.model_cfg, config.checkpoint,
                                    device=config.device, mode="train")
            except Exception as exc:
                raise RuntimeError(
                    f"build_sam2 failed for config={config.model_cfg!r}, checkpoint={config.checkpoint!r}. "
                    "Confirm SAM2 config root, checkpoint path, package revision, and compatible PyTorch."
                ) from exc
        self.backbone = sam2_model
        self.input_size = int(getattr(self.backbone, "image_size", 1024))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        enc_targets = config.encoder_target_modules or ["qkv", "attn.proj", "mlp.layers.0", "mlp.layers.1"]
        self.encoder_lora_layers: list[str] = []
        image_encoder = getattr(self.backbone, "image_encoder", None)
        if image_encoder is not None and enc_targets:
            # Encoder LoRA is deliberately a point estimate; posterior sampling
            # is restricted to the decoder in this research design.
            try:
                self.encoder_lora_layers = inject_deterministic_lora(
                    image_encoder, enc_targets, rank=config.rank_bank[-1],
                    alpha=config.alpha, dropout=config.dropout)
            except RuntimeError as exc:
                raise RuntimeError("No configured encoder LoRA modules matched the installed SAM2 image_encoder. "
                                   "Inspect named_modules() and update encoder_target_modules explicitly.") from exc
        # Checkpoint the official Hiera trunk during adapter training so that
        # intermediate block activations are recomputed during backward rather
        # than all retained at 1024x1024. This leaves feature dimensions and
        # SAM2 modules/weights unchanged. Fake test models without a Hiera trunk
        # safely skip this hardware-specific optimization.
        self.encoder_gradient_checkpointing_enabled = self._enable_encoder_gradient_checkpointing()

        decoder = getattr(self.backbone, "sam_mask_decoder", None)
        if decoder is None:
            raise AttributeError("Installed SAM2 model has no `sam_mask_decoder`; unsupported SAM2 API")
        try:
            self.decoder_lora_layers = inject_bayesian_lora(
                decoder, config.target_modules, rank=config.rank_bank[-1], alpha=config.alpha,
                dropout=config.dropout, prior_std=config.prior_std,
                init_posterior_std=config.init_posterior_std)
        except RuntimeError as exc:
            raise RuntimeError("No configured decoder LoRA modules matched SAM2 `sam_mask_decoder`. "
                               "Inspect the actual installed module tree; do not guess names.") from exc
        self.rank_controller = RankController(
            self.decoder_lora_layers,
            RankControllerConfig(list(config.rank_bank), gamma=config.attribution_gamma,
                                 learned_prior_weight=config.learned_prior_weight))
        self.register_buffer("_rgb_mean", torch.tensor(SAM2_RGB_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_rgb_std", torch.tensor(SAM2_RGB_STD).view(1, 3, 1, 1), persistent=False)
        self._last_attribution: dict[str, float] = {}
        self._last_selected_ranks: dict[str, int] = {}

    def _enable_encoder_gradient_checkpointing(self) -> bool:
        """Checkpoint Hiera blocks individually to cap backward recomputation peaks.

        A single checkpoint around the entire Hiera trunk can still OOM during
        backward: PyTorch must recompute a large section of the trunk while the
        decoder graph is resident. Checkpointing each official Hiera block keeps
        only block-boundary inputs and recomputes one block at a time. The actual
        SAM2 blocks, weights, input resolution, and feature geometry are unchanged.

        Tiny/fake backbones without a `.blocks` sequence retain the older
        whole-trunk fallback so CPU unit tests and compatible third-party models
        keep working.
        """
        image_encoder = getattr(self.backbone, "image_encoder", None)
        trunk = getattr(image_encoder, "trunk", None)
        if trunk is None or not callable(getattr(trunk, "forward", None)):
            return False
        if getattr(trunk, "_balora_gradient_checkpointing_enabled", False):
            return True
        model_ref = weakref.ref(self)
        blocks = getattr(trunk, "blocks", None)

        if blocks is not None and len(blocks) > 0:
            wrapped_count = 0
            for block in blocks:
                if getattr(block, "_balora_checkpoint_wrapped", False):
                    continue
                original_block_forward = block.forward

                def checkpointed_block_forward(*args, _original=original_block_forward, **kwargs):
                    owner = model_ref()
                    if owner is not None and owner.training and torch.is_grad_enabled():
                        return torch_checkpoint(_original, *args, use_reentrant=False, **kwargs)
                    return _original(*args, **kwargs)

                block.forward = checkpointed_block_forward
                block._balora_checkpoint_wrapped = True
                wrapped_count += 1
            trunk._balora_gradient_checkpointing_enabled = True
            trunk._balora_gradient_checkpointing_mode = "per_block"
            trunk._balora_gradient_checkpointing_block_count = len(blocks)
            self.encoder_gradient_checkpointing_mode = "per_block"
            self.encoder_gradient_checkpointing_block_count = len(blocks)
            return True

        # Fallback for fake/alternative trunks that expose no explicit block list.
        original_forward = trunk.forward

        def checkpointed_forward(x, *args, **kwargs):
            owner = model_ref()
            if owner is not None and owner.training and torch.is_grad_enabled():
                return torch_checkpoint(original_forward, x, *args,
                                        use_reentrant=False, **kwargs)
            return original_forward(x, *args, **kwargs)

        trunk.forward = checkpointed_forward
        trunk._balora_gradient_checkpointing_enabled = True
        trunk._balora_gradient_checkpointing_mode = "whole_trunk_fallback"
        self.encoder_gradient_checkpointing_mode = "whole_trunk_fallback"
        self.encoder_gradient_checkpointing_block_count = 0
        return True

    def train(self, mode: bool = True):
        """Keep pretrained SAM2 modules in eval mode while training only adapters.

        `nn.Module.train()` would otherwise activate stochastic-depth/dropout in
        frozen pretrained blocks. LoRA wrappers retain the requested mode so
        their configured dropout is still honored.
        """
        super().train(mode)
        self.backbone.eval()
        from models.bayesian_lora import DeterministicLoRALinear
        for module in self.backbone.modules():
            if isinstance(module, (DeterministicLoRALinear, BayesianLoRALinear)):
                module.train(mode)
        self.rank_controller.train(mode)
        return self

    def _decoder_modules(self) -> dict[str, BayesianLoRALinear]:
        return {name: layer for name, layer in self.backbone.sam_mask_decoder.named_modules()
                if isinstance(layer, BayesianLoRALinear)}

    def encode_image(self, image: torch.Tensor) -> SAM2EmbeddingBundle:
        """Encode grayscale BTCV (B,1,H,W) or RGB (B,3,H,W) in native model size."""
        if image.ndim != 4 or image.shape[1] not in (1, 3):
            raise ValueError(f"Expected Bx1xHxW or Bx3xHxW image, got {tuple(image.shape)}")
        if tuple(image.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"Input spatial size {tuple(image.shape[-2:])} does not match the official SAM2 model size "
                f"({self.input_size},{self.input_size}). Resize in the dataset/config; the adapter will not "
                "arbitrarily reshape or interpolate feature maps to conceal incompatibility."
            )
        raw = image.float().clamp(0, 1)
        if raw.shape[1] == 1:
            rgb = raw.repeat(1, 3, 1, 1)
        else:
            rgb = raw
        mean = self._rgb_mean.to(device=rgb.device, dtype=rgb.dtype)
        std = self._rgb_std.to(device=rgb.device, dtype=rgb.dtype)
        x = (rgb - mean) / std
        if not hasattr(self.backbone, "forward_image") or not hasattr(self.backbone, "_prepare_backbone_features"):
            raise AttributeError("SAM2 model lacks forward_image/_prepare_backbone_features from official SAM2Base API")
        # Evaluation needs gradients inside the decoder for uncertainty-based
        # attribution, but not through the frozen SAM2 encoder adapters. Avoid
        # building the full encoder autograd graph whenever the wrapper is in eval.
        # Mixed precision is encoder-only, including crop re-encoding during
        # evaluation/refinement. Bayesian attribution and losses stay FP32.
        encoder_amp = bool(getattr(self, 'encoder_amp', False) and x.device.type == 'cuda')
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=encoder_amp):
            if self.training and torch.is_grad_enabled():
                backbone_out = self.backbone.forward_image(x)
            else:
                with torch.no_grad():
                    backbone_out = self.backbone.forward_image(x)
        _, vision_feats, _, feat_sizes = self.backbone._prepare_backbone_features(backbone_out)
        if len(vision_feats) != len(feat_sizes) or not vision_feats:
            raise RuntimeError("SAM2 returned inconsistent visual features and feature sizes")
        if getattr(self.backbone, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + self.backbone.no_mem_embed
        batch = x.shape[0]
        feats_desc = []
        for seq_feature, spatial in zip(vision_feats[::-1], feat_sizes[::-1]):
            h, w = int(spatial[0]), int(spatial[1])
            if seq_feature.shape[0] != h*w or seq_feature.shape[1] != batch:
                raise RuntimeError(
                    f"SAM2 feature provenance mismatch: token tensor {tuple(seq_feature.shape)} does not "
                    f"match batch={batch}, spatial={h}x{w}. No resize workaround is applied."
                )
            feats_desc.append(seq_feature.permute(1, 2, 0).reshape(batch, -1, h, w).float())
        feats = feats_desc[::-1]
        image_embed, high_res_feats = feats[-1], feats[:-1]
        if image_embed.shape[1] != getattr(self.backbone, "hidden_dim", image_embed.shape[1]):
            # Don't assume 256 channels; use SAM2's own hidden dimension if present.
            raise RuntimeError(f"Unexpected SAM2 image embedding channels {image_embed.shape[1]}; model hidden_dim="
                               f"{getattr(self.backbone, 'hidden_dim', 'unknown')}")
        return SAM2EmbeddingBundle({"image_embed": image_embed, "high_res_feats": high_res_feats,
                                    "image_raw": raw, "input_size": (self.input_size, self.input_size),
                                    "feature_sizes": list(feat_sizes)})

    def _decode_once(self, embedding: dict[str, Any], points: torch.Tensor,
                     point_labels: torch.Tensor, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        embed = embedding["image_embed"]
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError(f"point_coords must be (B,N,2), got {tuple(points.shape)}")
        if point_labels.shape != points.shape[:2]:
            raise ValueError(f"point_labels shape {tuple(point_labels.shape)} must match points {tuple(points.shape[:2])}")
        if points.shape[0] != embed.shape[0]:
            raise ValueError(f"prompt batch {points.shape[0]} != embedding batch {embed.shape[0]}")
        prompt_encoder = getattr(self.backbone, "sam_prompt_encoder", None)
        decoder = getattr(self.backbone, "sam_mask_decoder", None)
        if prompt_encoder is None or decoder is None:
            raise AttributeError("SAM2 model must expose sam_prompt_encoder and sam_mask_decoder")
        sparse, dense = prompt_encoder(points=(points.to(device=embed.device, dtype=torch.float32),
                                               point_labels.to(device=embed.device, dtype=torch.int32)),
                                      boxes=None, masks=None)
        image_pe = prompt_encoder.get_dense_pe().to(device=embed.device, dtype=embed.dtype)
        high_res = embedding.get("high_res_feats")
        masks, _, _, _ = decoder(image_embeddings=embed, image_pe=image_pe,
                                 sparse_prompt_embeddings=sparse,
                                 dense_prompt_embeddings=dense, multimask_output=False,
                                 repeat_image=False, high_res_features=high_res)
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise RuntimeError(f"Expected single-mask SAM2 decoder output Bx1xHxW, got {tuple(masks.shape)}")
        size = output_size or tuple(embedding.get("image_raw", torch.empty(0, 1, *embed.shape[-2:])).shape[-2:])
        if tuple(masks.shape[-2:]) != tuple(size):
            masks = F.interpolate(masks, size=tuple(size), mode="bilinear", align_corners=False)
        return masks

    def decode(self, image_embedding: dict[str, Any], prompt_points: torch.Tensor,
               prompt_labels: torch.Tensor, k_samples: int = 4, sample: bool = True,
               ranks: dict[str, int] | None = None,
               output_size: tuple[int, int] | None = None) -> torch.Tensor:
        if k_samples < 1:
            raise ValueError("k_samples must be >=1")
        if sample and k_samples < 2:
            # Individual stochastic masks can be requested with K=1, but this
            # decoder API is used by uncertainty estimation; fail early.
            raise ValueError("k_samples must be >=2 when sample=True; epistemic uncertainty requires multiple samples")
        outputs = []
        with sampling_mode(self.backbone.sam_mask_decoder, sample=sample, ranks=ranks):
            for _ in range(k_samples):
                outputs.append(self._decode_once(image_embedding, prompt_points, prompt_labels, output_size))
        return torch.stack(outputs, dim=0)  # K,B,1,H,W

    def _attribution_scores(self, uncertainty: torch.Tensor, spatial_mask: torch.Tensor | None = None) -> dict[str, float]:
        """Compute numerically stable decoder-LoRA attribution scores.

        The attribution objective is the *mean* epistemic uncertainty rather
        than its pixel sum, so its scale does not grow with input resolution.
        Activations and gradients are promoted to float32 before multiplication
        to avoid fp16 product overflow during attribution.
        """
        import math

        modules = self._decoder_modules()
        outputs = [m.last_adapter_output for m in modules.values()]
        names = list(modules)
        valid = [(n, a) for n, a in zip(names, outputs) if a is not None and a.requires_grad]
        if not valid:
            raise RuntimeError("No live decoder LoRA activations are available for gradient attribution")

        signal_map = uncertainty
        if spatial_mask is not None:
            if spatial_mask.shape != signal_map.shape:
                raise ValueError(
                    f"attribution spatial mask {tuple(spatial_mask.shape)} != uncertainty {tuple(signal_map.shape)}"
                )
            signal_map = signal_map * spatial_mask.to(device=signal_map.device, dtype=signal_map.dtype)
        if not torch.isfinite(signal_map).all():
            raise FloatingPointError("Epistemic uncertainty contains NaN or infinity before attribution")

        # Normalize by the number of spatial elements; sum() makes gradients
        # scale with image area and can overflow at 1024x1024 resolution.
        signal = signal_map.float().mean()
        grads = torch.autograd.grad(
            signal, [activation for _, activation in valid],
            allow_unused=True, retain_graph=False
        )
        scores: dict[str, float] = {}
        bad_layers: list[str] = []
        for (name, activation), grad in zip(valid, grads):
            activation32 = activation.detach().float()
            if not torch.isfinite(activation32).all():
                bad_layers.append(f"{name}: non-finite activation")
                continue
            if grad is None:
                scores[name] = 0.0
                continue
            grad32 = grad.detach().float()
            if not torch.isfinite(grad32).all():
                bad_layers.append(f"{name}: non-finite gradient")
                continue
            contribution = (grad32 * activation32).abs()
            if not torch.isfinite(contribution).all():
                bad_layers.append(f"{name}: non-finite gradient-activation product")
                continue
            score = float(contribution.mean().item())
            if not math.isfinite(score):
                bad_layers.append(f"{name}: non-finite aggregated score")
                continue
            scores[name] = score

        for name in modules:
            scores.setdefault(name, 0.0)
        if bad_layers:
            raise FloatingPointError(
                "Non-finite decoder LoRA attribution values after float32 stabilization: "
                + "; ".join(bad_layers[:8])
            )
        bad_scores = [name for name, value in scores.items() if not math.isfinite(value)]
        if not scores or bad_scores:
            raise RuntimeError(
                "Gradient attribution yielded non-finite or empty scores; "
                f"bad layers={bad_scores[:8]}, score_count={len(scores)}"
            )
        return scores

    def select_ranks_from_uncertainty(self, embedding: dict[str, Any], points: torch.Tensor,
                                      point_labels: torch.Tensor, k_samples: int = 2,
                                      spatial_mask: torch.Tensor | None = None) -> tuple[dict[str, int], UncertaintyMaps]:
        if k_samples < 2:
            raise ValueError("k_samples >= 2 is required for uncertainty-driven rank selection")
        with torch.enable_grad():
            probe_logits = self.decode(embedding, points, point_labels, k_samples=k_samples, sample=True)
            uq = estimate_uncertainty(probe_logits.squeeze(2))
            scores = self._attribution_scores(uq.epistemic, spatial_mask=spatial_mask)
        ranks = self.rank_controller.select_ranks(scores)
        self._last_attribution, self._last_selected_ranks = scores, ranks
        uq = UncertaintyMaps(*(getattr(uq, name).detach() for name in (
            'mean_prob', 'total_uncertainty', 'expected_conditional_entropy', 'epistemic')))
        for module in self._decoder_modules().values():
            module.last_adapter_output = None
        return ranks, uq

    def decode_with_adaptive_rank(self, embedding: dict[str, Any], prompt_points: torch.Tensor,
                                  prompt_labels: torch.Tensor, k_samples: int = 4) -> tuple[torch.Tensor, UncertaintyMaps, dict[str, int]]:
        if k_samples < 2:
            raise ValueError("k_samples >= 2 is required for adaptive-rank inference")
        # Two posterior draws suffice for the probe statistic and keep the
        # attribution pass bounded; final prediction uses the requested K.
        ranks, probe_uq = self.select_ranks_from_uncertainty(
            embedding, prompt_points, prompt_labels, k_samples=2)
        logits = self.decode(embedding, prompt_points, prompt_labels,
                             k_samples=k_samples, sample=True, ranks=ranks)
        return logits, probe_uq, ranks

    def decode_region(self, image_embedding: dict[str, Any], prompt_point: tuple[int, int],
                      bbox: tuple[int, int, int, int], ranks: Dict[str, int],
                      k_samples: int = 4) -> torch.Tensor:
        """Re-encode an uncertainty crop using the actual source image tensor.

        Bounding boxes are inclusive (y0,x0,y1,x1) in image coordinates. The
        crop is resized to the official model input size; decoder features are
        then generated by the real SAM2 encoder API, not cropped from latent
        feature maps or fabricated via arbitrary interpolation.
        """
        raw = image_embedding.get("image_raw")
        if raw is None or raw.shape[0] != 1:
            raise ValueError("Regional refinement currently requires a single-image embedding bundle")
        y0,x0,y1,x1 = bbox
        h,w = raw.shape[-2:]
        y0=max(0,min(int(y0),h-1)); y1=max(y0,min(int(y1),h-1))
        x0=max(0,min(int(x0),w-1)); x1=max(x0,min(int(x1),w-1))
        crop = raw[..., y0:y1+1, x0:x1+1]
        crop_h,crop_w = crop.shape[-2:]
        crop_native = F.interpolate(crop, size=(self.input_size,self.input_size), mode="bilinear", align_corners=False)
        cropped_embedding = self.encode_image(crop_native)
        py,px = prompt_point
        points = torch.tensor([[[((px-x0) / max(crop_w-1,1)) * (self.input_size-1),
                                  ((py-y0) / max(crop_h-1,1)) * (self.input_size-1)]]],
                              device=raw.device, dtype=torch.float32)
        labels = torch.ones((1,1), device=raw.device, dtype=torch.int32)
        with torch.enable_grad():
            result = self.decode(cropped_embedding, points, labels, k_samples=k_samples,
                                 sample=True, ranks=ranks, output_size=(crop_h,crop_w))
        # Contract required by estimate_uncertainty/refine: (K,B,H,W), where
        # B=1. The decoder's public output is (K,B,1,H,W).
        return result.squeeze(2)

    def make_region_attribution_fn(self, embedding: dict[str, Any], point_labels: torch.Tensor,
                                   k_samples: int = 2):
        """Create region-scoped rank attribution using the shared gradient method."""
        def attribution_fn(epistemic_map: torch.Tensor, region: Any) -> dict[str, int]:
            y0,x0,y1,x1 = region.bbox
            points = torch.tensor([[[float((x0+x1)//2), float((y0+y1)//2)]]],
                                  device=epistemic_map.device, dtype=torch.float32)
            if epistemic_map.ndim == 2:
                region_mask = torch.zeros((1, *epistemic_map.shape), device=epistemic_map.device, dtype=torch.bool)
            else:
                region_mask = torch.zeros_like(epistemic_map, dtype=torch.bool)
            region_mask[:, y0:y1+1, x0:x1+1] = True
            ranks, _ = self.select_ranks_from_uncertainty(embedding, points, point_labels,
                                                         k_samples=k_samples, spatial_mask=region_mask)
            return ranks
        return attribution_fn

    def total_kl(self) -> torch.Tensor:
        return total_kl_divergence(self.backbone.sam_mask_decoder)
