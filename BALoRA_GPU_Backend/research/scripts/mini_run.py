"""Single-image SAM2 sanity check.

Run this AFTER installing `sam2` and downloading a checkpoint, BEFORE
starting any training run:

    python scripts/mini_run.py --config configs/default.yaml --image path/to/one_image.png

It does NOT train anything. It just confirms:
  1. `sam2` imports and the checkpoint loads.
  2. `inject_bayesian_lora` finds and wraps the expected layers (prints the
     list -- if it's empty or looks wrong, your `target_modules` names in
     the config don't match this sam2 version's module names).
  3. A forward pass (encode + decode) runs end-to-end and produces a mask
     of the expected shape.
  4. Running the decode K times produces K *different* masks (since the
     Bayesian posterior is stochastic) -- if all K outputs are identical,
     something is wrong with sampling (e.g. `sample=False` got hardcoded
     somewhere, or a random seed is being reset between calls).

If any of these fail, fix it here before touching training/train.py --
debugging a broken integration is far easier on one image than mid-training.
"""
from __future__ import annotations

import argparse

import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.confidence import estimate_uncertainty
from models.sam2_wrapper import Sam2BayesianLoRAConfig, Sam2BayesianLoRAModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n[1/4] Building model (this loads SAM2 + injects Bayesian LoRA)...")
    model_cfg = Sam2BayesianLoRAConfig(
        model_cfg=cfg["sam2"]["model_cfg"],
        checkpoint=cfg["sam2"]["checkpoint"],
        target_modules=cfg["lora"]["target_modules"],
        rank_bank=cfg["lora"]["rank_bank"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
        prior_std=cfg["bayesian"]["prior_std"],
        init_posterior_std=cfg["bayesian"]["init_posterior_std"],
        device=device,
    )
    model = Sam2BayesianLoRAModel(model_cfg).to(device)
    model.eval()

    n_wrapped = sum(1 for _ in model.rank_controller.layer_names)
    print(f"  -> {n_wrapped} layers wrapped with Bayesian LoRA:")
    for name in model.rank_controller.layer_names[:10]:
        print(f"     {name}")
    if n_wrapped > 10:
        print(f"     ... and {n_wrapped - 10} more")
    if n_wrapped == 0:
        raise RuntimeError(
            "0 layers wrapped -- target_modules in the config don't match "
            "this SAM2 version's module names. Print "
            "`model.backbone.named_modules()` and update "
            "configs/default.yaml's lora.target_modules."
        )

    print("\n[2/4] Loading and preprocessing test image...")
    image = Image.open(args.image).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
            transforms.ToTensor(),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    print(f"  -> image tensor shape: {tuple(image_tensor.shape)}")

    print("\n[3/4] Running encode + K-sample decode...")
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        h, w = image_tensor.shape[-2:]
        points = torch.tensor([[[w // 2, h // 2]]], dtype=torch.float32, device=device)
        point_labels = torch.ones(1, 1, dtype=torch.int64, device=device)

        k = cfg["bayesian"]["k_infer"]
        logits_k = model.decode(embedding, points, point_labels, k_samples=k, sample=True)
    print(f"  -> logits shape: {tuple(logits_k.shape)} (expected (K, 1, 1, H, W))")

    print("\n[4/4] Checking posterior samples actually differ...")
    logits_flat = logits_k.view(k, -1)
    pairwise_diffs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = (logits_flat[i] - logits_flat[j]).abs().mean().item()
            pairwise_diffs.append(diff)
    mean_diff = sum(pairwise_diffs) / len(pairwise_diffs)
    print(f"  -> mean pairwise |logit difference| across {k} samples: {mean_diff:.6f}")
    if mean_diff < 1e-6:
        print(
            "  WARNING: samples are (near-)identical. Either init_posterior_std "
            "is too small to see variation yet (expected early in training, "
            "before the posterior has learned meaningful variance), or "
            "sample=True isn't actually triggering stochastic sampling -- "
            "check BayesianLoRALinear.forward()."
        )
    else:
        print("  -> OK: posterior samples differ, stochasticity is working.")

    uq = estimate_uncertainty(logits_k.squeeze(2))
    print(
        f"\nUncertainty summary: mean_prob range=[{uq.mean_prob.min():.3f}, "
        f"{uq.mean_prob.max():.3f}], "
        f"epistemic mean={uq.epistemic.mean():.4f}, "
        f"aleatoric mean={uq.aleatoric.mean():.4f}"
    )
    print("\nAll sanity checks completed. Safe to proceed to a small training run.")


if __name__ == "__main__":
    main()
