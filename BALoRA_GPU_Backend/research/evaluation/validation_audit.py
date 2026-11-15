"""Checkpoint-only validation follow-up; never trains or evaluates test cases."""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Any
import zipfile

import torch
from torch.utils.data import DataLoader

from datasets.btcv_dataset import BTCVDataset, collate_btcv
from evaluation.evaluate import _model_fingerprint, run_evaluation
from models.refinement import RefinementConfig
from models.refinement_guarded import GuardedRefinementConfig
from training.train import build_model
from utils.checkpoint import load_checkpoint
from utils.runtime_budget import write_json
from utils.seed import set_seed


BUILD = "validation_refinement_audit_v1"
REVISION = "sam2_runtime_import_fix_v1_1"
REFERENCE_BEST_SHA256 = "8536f08e85dfd94eaa364513ee366b10b090e8948e00853e1a7bbd1f24113436"
METADATA = ("reproducibility_manifest.json", "selected_checkpoint.json", "session_state.json")
RESUME_FILES = ("legacy_progress.json", "guarded_progress.json", "legacy_validation.json",
                "guarded_validation.json", "validation_comparison.json", "validation_selection.json",
                "validation_per_organ.csv", "audit_session_state.json")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_saved_split(cfg: dict) -> dict[str, list[str]]:
    splits = cfg["experiment"]["split_case_ids"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Saved train/validation/test case identities are required")
    for key, expected in (("train", 19), ("validation", 5), ("test", 6)):
        if len(splits[key]) != expected or len(set(splits[key])) != expected:
            raise ValueError(f"The completed Fold-1 {key} split must have {expected} distinct cases")
    sets = [set(splits[k]) for k in ("train", "validation", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Saved case splits overlap")
    digest = hashlib.sha256(json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != cfg["experiment"]["split_hash"]:
        raise ValueError("Saved split hash does not match the saved case lists")
    return copy.deepcopy(splits)


def _zip_prefix(z: zipfile.ZipFile) -> str:
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate members in results ZIP")
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name:
            raise ValueError(f"Unsafe ZIP member: {name}")
    prefixes = [n[:-len("reproducibility_manifest.json")] for n in names
                if n.endswith("reproducibility_manifest.json")
                and n[:-len("reproducibility_manifest.json")]+"research_fold_01/best.pt" in names]
    if len(prefixes) != 1:
        raise ValueError("Results ZIP must contain one manifest and matching research_fold_01/best.pt")
    return prefixes[0]


def locate_results(input_root: str | Path, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    root = Path(input_root)
    folders = sorted({p.parent for p in root.rglob("selected_checkpoint.json")
                      if (p.parent/"reproducibility_manifest.json").is_file()
                      and (p.parent/"research_fold_01/best.pt").is_file()})
    if len(folders) == 1:
        return folders[0]
    if len(folders) > 1:
        raise ValueError("Several completed outputs found. Set RESULTS_INPUT to the intended results folder.")
    archives = sorted(p for p in root.rglob("*.zip")
                      if "balora" in p.name.lower() and "results" in p.name.lower())
    compatible = []
    for p in archives:
        with zipfile.ZipFile(p) as z:
            try:
                _zip_prefix(z)
            except ValueError:
                continue
        compatible.append(p)
    if len(compatible) != 1:
        raise FileNotFoundError("Attach the completed BALoRA results ZIP or its extracted output. "
                                "Set RESULTS_INPUT explicitly if several outputs are attached.")
    return compatible[0]


def prepare_reference(source: str | Path, local_inputs: str | Path,
                      expected_sha256: str = REFERENCE_BEST_SHA256) -> dict[str, Any]:
    """Read only checkpoint/identity/usage metadata, never held-out scores."""
    source, local = Path(source), Path(local_inputs)
    local.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        with zipfile.ZipFile(source) as z:
            prefix = _zip_prefix(z)
            for name in (*METADATA, "research_fold_01/best.pt"):
                member = prefix+name
                if member not in z.namelist():
                    raise FileNotFoundError(f"Required recovery file missing: {name}")
                dest = local/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
    else:
        for name in (*METADATA, "research_fold_01/best.pt"):
            src, dest = source/name, local/name
            if not src.is_file():
                raise FileNotFoundError(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
    manifest, selected, usage = [json.loads((local/name).read_text()) for name in METADATA]
    if not manifest.get("protocol_complete") or manifest.get("execution_status") != "FOLD1_PROTOCOL_COMPLETE":
        raise ValueError("This follow-up requires the completed Fold-1 results")
    checkpoint = local/"research_fold_01/best.pt"
    digest = sha256_file(checkpoint)
    if digest != selected["sha256"] or (expected_sha256 and digest != expected_sha256):
        raise ValueError("Selected checkpoint SHA-256 differs from the completed reference result")
    cfg = manifest["configuration"]
    splits = validate_saved_split(cfg)
    if not usage.get("closed_cleanly") or usage.get("execution_status") != "FOLD1_PROTOCOL_COMPLETE":
        raise ValueError("A clean completed session_state.json is needed for cumulative time accounting")
    prior_seconds = float(usage["accounted_runtime_seconds"])
    if not math.isfinite(prior_seconds) or prior_seconds < 0:
        raise ValueError("Invalid prior compute time")
    return {"source": str(source), "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
            "selected_epoch": selected["epoch"], "configuration": copy.deepcopy(cfg),
            "splits": splits, "prior_accounted_seconds": prior_seconds,
            "original_source_sha256": manifest.get("source_archive_sha256"),
            "original_fold1_status": manifest["execution_status"]}


def find_btcv_root(input_root: str | Path, explicit: str = "") -> Path:
    def matches(p):
        return p.is_dir() and any((p/k).is_dir() for k in ("img", "imagesTr", "images")) and \
               any((p/k).is_dir() for k in ("label", "labelsTr", "labels"))
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not matches(p):
            raise FileNotFoundError(f"BTCV_ROOT must contain paired image/label folders: {p}")
        return p
    candidates = sorted({p.resolve() for p in Path(input_root).rglob("Training") if matches(p)})
    if len(candidates) != 1:
        raise FileNotFoundError("Attach the original BTCV data and set BTCV_ROOT if several Training folders exist")
    return candidates[0]


def make_validation_loader(cfg: dict, btcv_root: str | Path, output_root: str | Path) -> DataLoader:
    splits = validate_saved_split(cfg)
    root = Path(output_root)
    split_path = root/"validation_cases.json"
    write_json(split_path, {"cases": splits["validation"], "split": "validation"})
    d = cfg["data"]
    dataset = BTCVDataset(str(btcv_root), str(split_path), num_classes=int(d["num_classes"]),
                          clip_len=1, train=False, image_size=int(d["image_size"]),
                          ct_window=tuple(d["ct_window"]), cache_dir=str(root/"raw_validation_cache"),
                          foreground_only=True)
    if set(dataset.case_ids) != set(splits["validation"]):
        raise ValueError("Validation dataset differs from the saved validation case list")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_btcv)


def _base_rows(path: Path) -> dict[tuple, dict]:
    payload = json.loads(path.read_text())
    return {(row["case_id"], row["slice_index"], row["organ_class"]): row["base_seg"]
            for row in payload["rows"]}


def compare_validation(model, loader, cfg: dict, output_root: str | Path, device: str,
                       deadline: float | None = None) -> dict:
    """One legacy and one guarded pass; paired baseline comes from both."""
    expected = set(cfg["experiment"]["split_case_ids"]["validation"])
    if set(getattr(loader.dataset, "case_ids", [])) != expected:
        raise ValueError("Only the saved validation volumes may enter this audit")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    r = cfg["refinement"]
    kwargs = dict(confidence_threshold=r["confidence_threshold"], max_iters=r["max_iters"],
                  crop_padding=r["crop_padding"], region_min_area=cfg["confidence"]["region_min_area"],
                  k_samples=cfg["bayesian"]["k_infer"])
    policies = {"legacy": RefinementConfig(**kwargs), "guarded": GuardedRefinementConfig(**kwargs)}
    fingerprint_before = _model_fingerprint(model)
    seed = int(cfg["experiment"]["seed"])+90001  # Same prompts as original model-selection validation.
    reports = {}
    for name, policy in policies.items():
        print(f"Validation only: {name}; {len(loader.dataset)} foreground slices; K={kwargs['k_samples']}", flush=True)
        report = run_evaluation(model, loader, device, k_infer=kwargs["k_samples"], refinement_config=policy,
                                compute_surface_distances=False, seed=seed, deadline=deadline,
                                include_unrefined=True, progress_path=root/f"{name}_progress.json",
                                progress_every=20)
        report["dataset_split"] = "validation"
        report["test_evaluated"] = False
        write_json(root/f"{name}_validation.json", report)
        reports[name] = report
        if not report["protocol"]["complete"]:
            break
    if _model_fingerprint(model) != fingerprint_before:
        raise RuntimeError("Model weights changed during inference-only audit")
    complete = len(reports) == 2 and all(r["protocol"]["complete"] for r in reports.values())
    comparison = {"build": BUILD, "complete": complete, "dataset_split": "validation",
                  "validation_cases": sorted(expected), "test_evaluated": False,
                  "training_updates": 0, "model_weights_unchanged": True,
                  "seed": seed, "k_infer": kwargs["k_samples"],
                  "model_fingerprint": fingerprint_before, "reports": reports,
                  "selection_rule": "Guarded macro-organ validation Dice must strictly exceed the paired baseline by >1e-6; otherwise use no refinement.",
                  "research_scope": "Exploratory validation follow-up after the original Fold-1 test was observed. No new independent test claim."}
    selection = {"complete": False, "selected_policy": None, "test_evaluated": False}
    if complete:
        legacy_base = _base_rows(root/"legacy_progress.json")
        guarded_base = _base_rows(root/"guarded_progress.json")
        if legacy_base.keys() != guarded_base.keys():
            raise ValueError("Policies did not evaluate the same slice/organ predictions")
        if not legacy_base:
            raise ValueError("No validation predictions were scored")
        for key in legacy_base:
            for metric in legacy_base[key]:
                a, b = legacy_base[key][metric], guarded_base[key][metric]
                if a is None and b is None:
                    continue
                if a is None or b is None or not math.isclose(a, b, rel_tol=0, abs_tol=1e-7):
                    raise ValueError(f"Initial predictions differ between policies at {key}/{metric}")
        def macro(per_organ):
            vals = [v["dice"]["mean"] for v in per_organ.values()]
            if not vals or any(v is None or not math.isfinite(v) for v in vals):
                raise ValueError("Incomplete per-organ validation Dice")
            return sum(vals)/len(vals)
        baseline = reports["legacy"]["same_checkpoint_no_refinement"]
        scores = {"none": macro(baseline["per_organ_segmentation"]),
                  "legacy": reports["legacy"]["macro_organ_dice"],
                  "guarded": reports["guarded"]["macro_organ_dice"]}
        selected = "guarded" if scores["guarded"] > scores["none"]+1e-6 else "none"
        selection = {"complete": True, "selected_policy": selected, "validation_macro_organ_dice": scores,
                     "rule": comparison["selection_rule"], "validation_cases": sorted(expected),
                     "seed": seed, "k_infer": kwargs["k_samples"], "test_evaluated": False,
                     "model_fingerprint": fingerprint_before,
                     "guarded_policy": asdict(policies["guarded"])}
        comparison["paired_initial_predictions_verified"] = True
        comparison["n_scored_slice_organ_predictions"] = len(legacy_base)
        with (root/"validation_per_organ.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["organ", "no_refinement_dice", "legacy_dice", "guarded_dice", "guarded_minus_none"])
            for organ in sorted(baseline["per_organ_segmentation"]):
                base = baseline["per_organ_segmentation"][organ]["dice"]["mean"]
                old = reports["legacy"]["per_organ_segmentation"][organ]["dice"]["mean"]
                new = reports["guarded"]["per_organ_segmentation"][organ]["dice"]["mean"]
                writer.writerow([organ, base, old, new, new-base])
    comparison["selection"] = selection
    write_json(root/"validation_comparison.json", comparison)
    write_json(root/"validation_selection.json", selection)
    return comparison


def prepare_audit_resume(input_root: Path, output_root: Path, reference: dict,
                         explicit_root: str = "") -> float:
    candidates = []
    possible = [Path(explicit_root)/"audit_session_state.json"] if explicit_root else \
               [output_root/"audit_session_state.json", *input_root.rglob("audit_session_state.json")]
    for path in possible:
        if not path.is_file():
            continue
        state = json.loads(path.read_text())
        if state.get("checkpoint_sha256") == reference["checkpoint_sha256"] and state.get("build") == BUILD:
            candidates.append((path, state))
    if len(candidates) > 1:
        raise ValueError("Multiple audit outputs found; set AUDIT_RESUME_ROOT to the intended extracted output")
    prior = reference["prior_accounted_seconds"]
    if candidates:
        path, state = candidates[0]
        if state.get("closed_cleanly"):
            used = float(state["accounted_runtime_seconds"])
        else:
            used = float(state["prior_accounted_seconds"])+12*3600
        if not math.isfinite(used) or used < 0:
            raise ValueError("Invalid audit usage ledger")
        prior = max(prior, used)
        for name in RESUME_FILES:
            src, dest = path.parent/name, output_root/name
            if src.is_file() and src.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
    return prior


def run_audit(reference: dict, btcv_root: Path, output_root: Path, sam2_checkpoint: Path,
              sam2_commit: str, input_root: Path, *, notebook_start: float | None = None,
              work_hours: float = 3, total_hours: float = 29.5, audit_resume_root: str = "",
              minimum_prior_hours: float = 0) -> dict:
    """GPU entry point. All cumulative compute is recorded, including export."""
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle CUDA GPU for the real SAM2 validation audit")
    if not 0 < work_hours <= 11.5 or not 0 < total_hours <= 29.5:
        raise ValueError("Work cap must be <=11.5h and cumulative cap <=29.5h")
    if not math.isfinite(float(minimum_prior_hours)) or minimum_prior_hours < 0:
        raise ValueError("Minimum prior hours must be finite and non-negative")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time() if notebook_start is None else float(notebook_start)
    prior = max(prepare_audit_resume(input_root, output_root, reference, audit_resume_root),
                float(minimum_prior_hours)*3600)
    # A failed attempt may already have closed its ledger in this kernel.
    # Keep the original session deadline while counting subsequent time once.
    accounting_start = started
    previous_ledger = output_root/"audit_session_state.json"
    if previous_ledger.is_file():
        previous = json.loads(previous_ledger.read_text())
        if (previous.get("build") == BUILD and
                previous.get("checkpoint_sha256") == reference["checkpoint_sha256"] and
                previous.get("closed_cleanly")):
            finished = float(previous.get("finished_unix", started))
            if not math.isfinite(finished) or finished > time.time():
                raise ValueError("Invalid prior audit finish time")
            accounting_start = max(started, finished)
    work_deadline = min(started+work_hours*3600, started+11.5*3600,
                        accounting_start+total_hours*3600-prior-900)  # 15 min export reserve.
    if work_deadline <= time.time():
        raise RuntimeError("No compute remains inside the cumulative/session work budget")
    state = {"build": BUILD, "revision": REVISION, "checkpoint_sha256": reference["checkpoint_sha256"],
             "session_start_unix": started, "prior_accounted_seconds": prior,
             "accounting_start_unix": accounting_start,
             "closed_cleanly": False, "session_limit_hours": 12,
             "work_deadline_unix": work_deadline, "total_budget_hours": total_hours}
    ledger = output_root/"audit_session_state.json"
    write_json(ledger, state)
    status = "FAILED"
    result = None
    export_names = []
    try:
        cfg = copy.deepcopy(reference["configuration"])
        expected_sam = cfg["sam2"]
        actual_base_hash = sha256_file(sam2_checkpoint)
        if actual_base_hash != expected_sam["checkpoint_sha256"] or sam2_commit != expected_sam["resolved_commit"]:
            raise ValueError("Frozen SAM2 checkpoint/source differs from the trained reference")
        cfg["sam2"]["checkpoint"] = str(sam2_checkpoint)
        cfg["experiment"]["output_dir"] = str(output_root)
        cfg["data"]["root_dir"] = str(btcv_root)
        cfg["runtime"] = {"validation_only": True, "deadline": work_deadline}
        write_json(output_root/"audit_reference.json", {k: v for k, v in reference.items() if k != "configuration"})
        write_json(output_root/"audit_config.json", cfg)
        set_seed(cfg["experiment"]["seed"], deterministic=True)
        loader = make_validation_loader(cfg, btcv_root, output_root)
        print("Validation cases:", loader.dataset.case_ids, "| foreground slices:", len(loader.dataset), flush=True)
        print("Prior accounted hours:", round(prior/3600, 4), "| work limit hours:", work_hours, flush=True)
        model = build_model(cfg, "cuda:0")
        load_checkpoint(model, None, reference["checkpoint"], map_location="cpu", restore_rng=False)
        model.eval()
        result = compare_validation(model, loader, cfg, output_root, "cuda:0", work_deadline)
        result["checkpoint_sha256"] = reference["checkpoint_sha256"]
        result["selected_epoch"] = reference["selected_epoch"]
        result["gpu"] = torch.cuda.get_device_name(0)
        result["torch"] = torch.__version__
        write_json(output_root/"validation_comparison.json", result)
        status = "VALIDATION_AUDIT_COMPLETE" if result["complete"] else "PARTIAL_VALIDATION_AUDIT_RESUME_REQUIRED"
        if result["complete"]:
            print(json.dumps(result["selection"], indent=2), flush=True)
        else:
            print("Saved partial validation. Attach this output plus the original results in the next session.", flush=True)
        export_names = [name for name in (*RESUME_FILES, "audit_reference.json", "audit_config.json", "validation_cases.json")
                        if name != "audit_session_state.json" and (output_root/name).is_file()]
        with zipfile.ZipFile(output_root/"BALoRA_validation_audit_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in export_names:
                z.write(output_root/name, output_root.name+"/"+name)
    finally:
        state.update(closed_cleanly=True, accounted_runtime_seconds=prior+max(0, time.time()-accounting_start),
                     finished_unix=time.time(), execution_status=status)
        write_json(ledger, state)
        archive = output_root/"BALoRA_validation_audit_results.zip"
        if export_names and archive.is_file():
            with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as z:
                z.write(ledger, output_root.name+"/"+ledger.name)
        print("Audit status:", status, "| cumulative accounted hours:", round(state["accounted_runtime_seconds"]/3600, 4), flush=True)
    return result
