"""Environment and artifact manifest helpers; no metrics are synthesized."""
from __future__ import annotations
import hashlib, importlib.metadata, json, platform, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
from config import canonical_config_hash

def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(chunk_size),b""):
            h.update(chunk)
    return h.hexdigest()

def _git_commit(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:
        return None

def build_reproducibility_manifest(cfg: dict[str,Any], project_root: str | Path,
                                  dataset_cases: list[str] | None = None,
                                  split_manifest: dict[str,Any] | None = None,
                                  sam2_root: str | Path | None = None,
                                  checkpoint_path: str | Path | None = None,
                                  execution_status: str = "CONFIGURED_NOT_TRAINED") -> dict[str,Any]:
    project_root=Path(project_root).resolve()
    versions={}
    for package in ("torch","torchvision","numpy","scipy","nibabel","PyYAML","pytest","tensorboard"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    gpu=None
    if torch.cuda.is_available():
        props=torch.cuda.get_device_properties(0)
        gpu={"name":props.name,"total_memory_bytes":int(props.total_memory),"capability":list(torch.cuda.get_device_capability(0))}
    manifest={
      "project":"BALoRA", "version":"0.1.0", "created_utc":datetime.now(timezone.utc).isoformat(),
      "project_git_commit":_git_commit(project_root), "python":platform.python_version(),
      "platform":platform.platform(), "torch":torch.__version__, "cuda_runtime":torch.version.cuda,
      "cuda_available":torch.cuda.is_available(), "gpu":gpu, "package_versions":versions,
      "sam2": {"repository":"https://github.com/facebookresearch/sam2.git",
               "resolved_commit":_git_commit(sam2_root) if sam2_root else None,
               "model_cfg":cfg.get("sam2",{}).get("model_cfg"),
               "checkpoint_path":str(checkpoint_path) if checkpoint_path else cfg.get("sam2",{}).get("checkpoint"),
               "checkpoint_sha256":sha256_file(checkpoint_path) if checkpoint_path and Path(checkpoint_path).is_file() else None},
      "dataset":{"source":cfg.get("data",{}).get("root_dir"),"case_ids":sorted(dataset_cases or []),
                 "n_cases":len(dataset_cases or []),"split_hash":(split_manifest or {}).get("split_hash")},
      "configuration":cfg,"configuration_sha256":canonical_config_hash(cfg),
      "model_design":{"encoder_lora":"point-estimate","decoder_lora":"Bayesian factorized Gaussian posterior",
                      "adaptive_rank":"gradient-attribution-driven discrete rank bank","refinement":"inference-time uncertainty-triggered crop re-encoding"},
      "execution_status":execution_status,
      "research_metrics_generated":False,
      "known_limitations":["Real SAM2/GPU/BTCV tests are required in Kaggle before any research claims.",
                            "Evaluation currently uses one positive ground-truth-derived point per scored organ slice.",
                            "This script does not perform AMOS22 cross-dataset evaluation."]}
    return manifest

def write_manifest(manifest: dict[str,Any], path: str | Path) -> Path:
    dest=Path(path); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str),encoding="utf-8")
    return dest
