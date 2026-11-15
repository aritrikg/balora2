"""Atomic checkpoints including optimizer/scheduler/RNG state for resumable runs."""
from __future__ import annotations
import os, random, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch


def _rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available(): state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    if "python" in state: random.setstate(state["python"])
    if "numpy" in state: np.random.set_state(state["numpy"])
    if "torch_cpu" in state: torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        # A Kaggle resume can expose fewer GPUs than the saving session.
        # Restore matching devices; device 0 is the declared training device.
        states = state['torch_cuda'][:torch.cuda.device_count()]
        torch.cuda.set_rng_state_all(states)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                    epoch: int, path: str, extra: dict[str, Any] | None = None,
                    scheduler: Any = None, adapter_only: bool = False) -> None:
    dest = Path(path); dest.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if adapter_only:
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        state = {k: v.detach().cpu() for k, v in state.items() if k in names}
    payload: dict[str, Any] = {"epoch": int(epoch), "model_state_dict": state,
                               "checkpoint_format": "adapters_v1" if adapter_only else "full_v1",
                               "rng_state": _rng_state()}
    if optimizer is not None: payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None: payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra: payload.update(extra)
    # Write to a temporary file in the destination directory then atomically replace.
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent)); os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                    path: str, map_location: str | torch.device = "cpu",
                    scheduler: Any = None, restore_rng: bool = True) -> int:
    src = Path(path)
    if not src.is_file(): raise FileNotFoundError(f"Checkpoint not found: {src}")
    try:
        payload = torch.load(src, map_location=map_location, weights_only=False)
    except TypeError:  # older PyTorch without weights_only arg
        payload = torch.load(src, map_location=map_location)
    if payload.get('checkpoint_format') == 'adapters_v1':
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(payload['model_state_dict']) != expected:
            raise ValueError('Adapter checkpoint does not match the model trainable parameter names')
        saved_sam = payload.get('config', {}).get('sam2', {})
        if saved_sam:
            current_cfg = getattr(model, '_run_config', None)
            if current_cfg is not None:
                for key in ('checkpoint_sha256', 'resolved_commit'):
                    if saved_sam.get(key) and saved_sam[key] != current_cfg.get('sam2', {}).get(key):
                        raise ValueError(f'Frozen SAM2 identity mismatch for {key}')
        model.load_state_dict(payload['model_state_dict'], strict=False)
    else:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng and "rng_state" in payload:
        # torch.load(map_location='cuda') also moves CPU RNG byte tensors.
        state = payload['rng_state']
        if 'torch_cpu' in state: state['torch_cpu'] = state['torch_cpu'].cpu()
        if 'torch_cuda' in state: state['torch_cuda'] = [v.cpu() for v in state['torch_cuda']]
        _restore_rng(state)
    # Checkpoints store the zero-based epoch that just completed. Resume at
    # the next epoch to avoid repeating optimizer updates from that epoch.
    return int(payload.get("epoch", -1)) + 1
