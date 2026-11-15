"""BTCV discovery, validation, and deterministic volume-level splitting."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import Any

NIFTI_SUFFIXES = (".nii", ".nii.gz")
_IMAGE_DIRS = {"img", "image", "images", "imagestr", "images_ts", "imagestr"}
_LABEL_DIRS = {"label", "labels", "labelstr", "labels_ts", "seg", "segs", "groundtruth"}


def strip_nifti_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return name


def _role_for_path(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    for part in reversed(parts[:-1]):
        base = strip_nifti_suffix(part).lower()
        if base in _IMAGE_DIRS:
            return "image"
        if base in _LABEL_DIRS:
            return "label"
    stem = strip_nifti_suffix(path.name).lower()
    if re.match(r"^(img|image)[_-]?\d+", stem):
        return "image"
    if re.match(r"^(label|labels|seg)[_-]?\d+", stem):
        return "label"
    return None


def _canonical_id_from_component(component: str, role: str | None) -> str | None:
    stem = strip_nifti_suffix(component)
    # Covers standard filenames and case-directory names such as img0001.nii.
    m = re.fullmatch(r"(?:img|image|label|labels|seg|volume)[_-]?([A-Za-z0-9]+)", stem, re.I)
    if m:
        suffix = m.group(1)
        # Canonicalize the common numeric Synapse BTCV distribution.
        if suffix.isdigit():
            suffix = suffix.zfill(4)
        return f"img{suffix}" if suffix.isdigit() or role in {"image", "label"} else f"case{suffix}"
    # Generic IDs must be identical in image and label filenames.
    m = re.fullmatch(r"(case|subject|patient|scan)[_-]?([A-Za-z0-9]+)", stem, re.I)
    if m:
        return f"{m.group(1).lower()}{m.group(2)}"
    # Nested Kaggle files may have a randomized tmp filename. Ignore it and
    # search other path components for the actual case directory.
    return None


def canonical_case_id(path: str | Path, role: str | None = None) -> str | None:
    p = Path(path)
    inferred_role = role or _role_for_path(p)
    for component in reversed(p.parts):
        case_id = _canonical_id_from_component(component, inferred_role)
        if case_id:
            # A label-prefixed ID above is deliberately normalized to imgNNNN.
            return case_id
    stem = strip_nifti_suffix(p.name)
    if stem and not stem.lower().startswith("tmp-"):
        return stem
    return None


def discover_btcv_pairs(root_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Recursively find and pair image/label NIfTI files by canonical case ID.

    Supports Synapse/Kaggle `Training/img/img0001.nii/tmp-*.nii` and
    `Training/label/label0001.nii/tmp-*.nii`, as well as `imagesTr/labelsTr`.
    Duplicate IDs and unclassified NIfTIs are reported as errors rather than
    silently resolving filesystem-order conflicts.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"BTCV root does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(NIFTI_SUFFIXES))
    if not files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found below {root}")
    pairs: dict[str, dict[str, Path]] = {}
    unclassified: list[str] = []
    duplicates: list[str] = []
    for path in files:
        role = _role_for_path(path)
        if role is None:
            unclassified.append(str(path))
            continue
        cid = canonical_case_id(path, role)
        if cid is None:
            unclassified.append(str(path))
            continue
        slot = "image" if role == "image" else "label"
        entry = pairs.setdefault(cid, {})
        if slot in entry:
            duplicates.append(f"{cid} {slot}: {entry[slot]} and {path}")
        else:
            entry[slot] = path
    if unclassified:
        raise ValueError("Could not classify these NIfTI files as image or label; inspect root/layout:\n" + "\n".join(unclassified[:20]))
    if duplicates:
        raise ValueError("Duplicate NIfTI case IDs found:\n" + "\n".join(duplicates[:20]))
    missing = [f"{cid}: missing {slot}" for cid, entry in sorted(pairs.items()) for slot in ("image", "label") if slot not in entry]
    if missing:
        raise ValueError("Image/label pairing incomplete; refusing to drop cases:\n" + "\n".join(missing[:30]))
    return dict(sorted(pairs.items()))


def validate_btcv_pairs(pairs: dict[str, dict[str, Path]], num_classes: int = 14,
                        scan_label_values: bool = True) -> dict[str, Any]:
    """Validate headers and (optionally) observed label values using nibabel."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Install nibabel before validating NIfTI volumes: pip install nibabel") from exc
    report: dict[str, Any] = {"n_cases": len(pairs), "cases": {}, "errors": []}
    for cid, pair in pairs.items():
        im = nib.load(str(pair["image"]))
        lab = nib.load(str(pair["label"]))
        rec: dict[str, Any] = {"image_path": str(pair["image"]), "label_path": str(pair["label"]),
                               "image_shape": list(im.shape), "label_shape": list(lab.shape)}
        if len(im.shape) != 3 or len(lab.shape) != 3:
            report["errors"].append(f"{cid}: expected 3D image/label volumes")
        if tuple(im.shape) != tuple(lab.shape):
            report["errors"].append(f"{cid}: shape mismatch {im.shape} vs {lab.shape}")
        affine_equal = bool(__import__("numpy").allclose(im.affine, lab.affine, rtol=0.0, atol=1e-4))
        rec["affine_equal"] = affine_equal
        if not affine_equal:
            report["errors"].append(f"{cid}: image/label affines differ; orientation resampling is not implicit")
        if scan_label_values:
            np = __import__("numpy")
            vals = np.unique(lab.get_fdata(dtype=np.float32))
            non_integer = vals[np.abs(vals - np.round(vals)) > 1e-3]
            out_of_range = vals[(vals < 0) | (vals >= num_classes)]
            rec["label_values"] = [int(round(float(v))) for v in vals if abs(float(v)-round(float(v)) <= 1e-3)]
            rec["non_integer_label_values"] = [float(v) for v in non_integer[:20]]
            rec["out_of_range_label_values"] = [float(v) for v in out_of_range[:20]]
            if non_integer.size:
                report["errors"].append(f"{cid}: non-integer label values: {non_integer[:20].tolist()}")
            if out_of_range.size:
                report["errors"].append(f"{cid}: label values outside [0,{num_classes-1}]: {out_of_range[:20].tolist()}")
        report["cases"][cid] = rec
    report["valid"] = not report["errors"]
    if report["errors"]:
        raise ValueError("BTCV validation failed:\n" + "\n".join(report["errors"][:50]))
    return report


def split_manifest_hash(manifest: dict[str, Any]) -> str:
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def make_fold_split(case_ids: list[str], n_folds: int = 5, fold: int = 0,
                    seed: int = 42, val_fraction: float = 0.15) -> dict[str, Any]:
    """Deterministic volume-level outer-fold split plus inner validation split."""
    import random
    ids = sorted(set(case_ids))
    if len(ids) != len(case_ids):
        raise ValueError("case_ids contains duplicates")
    if n_folds < 2 or not 0 <= fold < n_folds:
        raise ValueError("Require n_folds>=2 and fold in [0,n_folds)")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    rng = random.Random(seed)
    shuffled = ids.copy(); rng.shuffle(shuffled)
    fold_buckets = [shuffled[i::n_folds] for i in range(n_folds)]
    test = sorted(fold_buckets[fold])
    remaining = sorted(set(ids) - set(test))
    rng_inner = random.Random(seed + 1009 + fold)
    rng_inner.shuffle(remaining)
    n_val = max(1, int(round(len(remaining) * val_fraction))) if len(remaining) > 2 else 1
    val = sorted(remaining[:n_val])
    train = sorted(remaining[n_val:])
    if not train or not val or not test:
        raise ValueError("Split produced an empty train/validation/test partition")
    if (set(train) & set(val)) or (set(train) & set(test)) or (set(val) & set(test)):
        raise AssertionError("Leakage detected between volume-level splits")
    return {"seed": seed, "fold": fold, "n_folds": n_folds,
            "train": train, "validation": val, "test": test,
            "split_hash": split_manifest_hash({"train": train, "validation": val, "test": test})}


def write_fold_splits(split: dict[str, Any], output_dir: str | Path) -> list[Path]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, fname in (("train", "btcv_train.json"), ("validation", "btcv_val.json"), ("test", "btcv_test.json")):
        payload = {"cases": split[key], "seed": split["seed"], "fold": split["fold"],
                   "n_folds": split["n_folds"], "split_hash": split["split_hash"]}
        path = out / fname
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths.append(path)
    return paths
