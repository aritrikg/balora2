"""Run the archived SAM2/LoRA implementations on one uploaded CT slice."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from setup_models import BASE, CHECKOUT, IDENTITY, ROOT, WEIGHTS, digest

METHODS = {
    "balora": "Gaussian Decoder LoRA (BALoRA)",
    "deterministic": "Deterministic LoRA",
    "frozen": "Frozen SAM2",
}
SIZE = 1024


def model_point(point, shape):
    y_size, x_size = shape
    x, y = point
    scaled = [(x + .5) * SIZE / x_size - .5, (y + .5) * SIZE / y_size - .5]
    return [min(SIZE - 1, max(0, coordinate)) for coordinate in scaled]


def load_model(method):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the CUDA-enabled PyTorch build on an NVIDIA GPU computer.")
    if not BASE.is_file() or not CHECKOUT.is_dir():
        raise RuntimeError("Run setup_models.py with the completed Fold 1 archive first.")
    if digest(BASE) != IDENTITY["backbone_sha256"]:
        raise ValueError("The SAM2 backbone checksum does not match the study.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
    if head != IDENTITY["sam2_commit"]:
        raise ValueError("The SAM2 checkout differs from the pinned study commit.")

    weight = WEIGHTS / f"{method}.pt"
    if method in IDENTITY["weight_sha256"] and (not weight.is_file() or digest(weight) != IDENTITY["weight_sha256"][method]):
        raise ValueError("The selected adapter is missing or differs from the verified archive. Run setup_models.py.")
    for path in (ROOT / "research", CHECKOUT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import sam2
    if not sam2.__file__ or not Path(sam2.__file__).resolve().is_relative_to(CHECKOUT.resolve()):
        raise RuntimeError("Another SAM2 package was loaded. Restart the local app after model setup.")

    cfg = json.loads((ROOT / "configs" / f"{method}.json").read_text())
    cfg["sam2"]["checkpoint"] = str(BASE)
    if method == "balora":
        from training.train import build_model
        model = build_model(cfg, "cuda:0")
    elif method == "deterministic":
        from models.deterministic_sam2 import DeterministicSAM2
        model = DeterministicSAM2(cfg, "cuda:0")
    else:
        from models.frozen_sam2 import FrozenSAM2
        model = FrozenSAM2(cfg, "cuda:0")
    if method in IDENTITY["weight_sha256"]:
        from utils.checkpoint import load_checkpoint
        load_checkpoint(model, None, str(weight), map_location="cpu", restore_rng=False)
    return model.eval()


def segment(model, method, gray, point):
    import torch
    from torch.nn.functional import interpolate

    img = torch.from_numpy(np.ascontiguousarray(gray)).unsqueeze(0).unsqueeze(0).to("cuda:0")
    points = torch.tensor([[model_point(point, gray.shape)]], dtype=torch.float32, device=img.device)
    labels = torch.ones((1, 1), dtype=torch.int32, device=img.device)
    with torch.no_grad():
        img = interpolate(img, (SIZE, SIZE), mode="bilinear", align_corners=False)
        embedding = model.encode_image(img)
        if method == "balora":
            from models.confidence import estimate_uncertainty
            logits, _, _ = model.decode_with_adaptive_rank(embedding, points, labels, k_samples=4)
            probability = estimate_uncertainty(logits.squeeze(2)).mean_prob[0]
        else:
            probability = model.decode_one(embedding, points, labels).sigmoid().reshape(SIZE, SIZE)
        mask = probability >= 0.5
        mask = interpolate(mask[None, None].float(), gray.shape, mode="nearest")[0, 0]
    return mask.bool().cpu().numpy()
