"""BTCV/Synapse 2D-slice clip dataset with robust recursive NIfTI discovery."""
from __future__ import annotations
from dataclasses import dataclass
import json
import hashlib, os, shutil, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.splits import discover_btcv_pairs

try:
    import nibabel as nib
except ImportError:  # availability checked at construction
    nib = None

BTCV_ORGAN_NAMES = [
    "background", "spleen", "right_kidney", "left_kidney", "gallbladder",
    "esophagus", "liver", "stomach", "aorta", "inferior_vena_cava",
    "portal_vein_splenic_vein", "pancreas", "right_adrenal_gland", "left_adrenal_gland",
]

@dataclass
class BTCVSample:
    image: torch.Tensor           # (T,1,H,W), intensity in [0,1]
    label: torch.Tensor           # (T,C,H,W), one-hot masks
    case_id: str
    slice_indices: list[int]
    valid_slices: torch.Tensor    # (T,), False only for collate padding

class BTCVDataset(Dataset):
    """Read BTCV volumes without dropping cases or silently clamping labels.

    split_file is a JSON object with a `cases` list. Inputs can be nested
    Synapse/Kaggle NIfTIs or imagesTr/labelsTr folders. This is 2D slice
    segmentation; `clip_len` is a storage/batching convenience, not temporal
    SAM2 video training. No orientation resampling or spatial augmentation is
    performed implicitly.
    """
    def __init__(self, root_dir: str, split_file: str, num_classes: int = 14,
                 clip_len: int = 1, train: bool = True, image_size: int = 1024,
                 ct_window: tuple[float, float] = (-175.0, 250.0),
                 cache_dir: str | None = None, foreground_only: bool = False) -> None:
        if nib is None:
            raise ImportError("nibabel is required: pip install nibabel")
        self.root_dir = str(Path(root_dir).expanduser().resolve())
        self.num_classes = int(num_classes)
        self.clip_len = int(clip_len)
        self.train = bool(train)
        self.image_size = int(image_size)
        self.ct_window = (float(ct_window[0]), float(ct_window[1]))
        self.cache_dir = Path(cache_dir or os.environ['BALORA_CACHE_DIR']) if (cache_dir or os.environ.get('BALORA_CACHE_DIR')) else None
        self._volume_cache = {}
        self.foreground_only = bool(foreground_only)
        if self.num_classes != len(BTCV_ORGAN_NAMES):
            # Configurable class counts are allowed only if data labels fit; use
            # the standard 14-channel contract by default and report names if so.
            if self.num_classes < 2:
                raise ValueError("num_classes must include background and at least one organ")
        if self.clip_len <= 0 or self.image_size <= 0:
            raise ValueError("clip_len and image_size must be positive")
        if not self.ct_window[0] < self.ct_window[1]:
            raise ValueError("ct_window must be (low, high) with low < high")
        split_path = Path(split_file)
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        self.case_ids = list(split_payload.get("cases", []))
        if not self.case_ids:
            raise ValueError(f"Split file has no 'cases' entries: {split_path}")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError(f"Duplicate case IDs in {split_path}")
        all_pairs = discover_btcv_pairs(self.root_dir)
        missing = sorted(set(self.case_ids) - set(all_pairs))
        if missing:
            raise FileNotFoundError(f"Split IDs not found under {self.root_dir}: {missing}")
        self.pairs = {cid: all_pairs[cid] for cid in self.case_ids}
        self._shapes: dict[str, tuple[int, int, int]] = {}
        self._clip_index: list[tuple[str, int, int]] = []
        for cid in self.case_ids:
            im = nib.load(str(self.pairs[cid]["image"]))
            lab = nib.load(str(self.pairs[cid]["label"]))
            if len(im.shape) != 3 or len(lab.shape) != 3:
                raise ValueError(f"{cid}: expected 3D NIfTI, got {im.shape} and {lab.shape}")
            if tuple(im.shape) != tuple(lab.shape):
                raise ValueError(f"{cid}: image/label shapes differ: {im.shape} vs {lab.shape}")
            if not np.allclose(np.asarray(im.affine), np.asarray(lab.affine), rtol=0, atol=1e-4):
                raise ValueError(f"{cid}: image/label affines differ; refusing implicit orientation resampling")
            self._shapes[cid] = tuple(int(x) for x in im.shape)
            n_slices = int(im.shape[2])
            # Non-overlapping windows: no duplicate validation slices at the end.
            for start in range(0, n_slices, self.clip_len):
                self._clip_index.append((cid, start, min(n_slices, start + self.clip_len)))
        self._observed_labels: dict[str, list[int]] = {}
        self.total_clips_before_filter = len(self._clip_index)
        self.background_clips_removed = 0
        if self.foreground_only:
            # The positive-prompt objective already skips these clips. Filter
            # before encoding, retaining EVERY slice containing ANY organ.
            kept = []
            for cid in self.case_ids:
                _, lab = self._read_volume(cid)
                lab = np.asanyarray(lab)
                self._validate_labels(lab, cid)
                present = np.any(lab > 0, axis=(0, 1))
                kept.extend((c, a, b) for c, a, b in self._clip_index
                            if c == cid and bool(present[a:b].any()))
            self._clip_index = kept
            self.background_clips_removed = self.total_clips_before_filter - len(kept)
            if not kept:
                raise ValueError('Split contains no organ-positive slices')

    def _validate_labels(self, lab, cid):
        vals = np.unique(lab)
        bad = vals[(~np.isfinite(vals)) | (np.abs(vals - np.round(vals)) > 1e-3) |
                   (vals < 0) | (vals >= self.num_classes)]
        if bad.size:
            raise ValueError(f'{cid}: unexpected/non-integer labels {bad[:20].tolist()}')

    def _read_volume(self, cid):
        """Decompress once, then read slice ranges from raw .npy memory maps.

        Raw CT values and label geometry are preserved. Cache keys include
        source identity, size and mtime; interrupted writes are never reused.
        """
        if cid in self._volume_cache:
            return self._volume_cache[cid]
        pair = self.pairs[cid]
        if self.cache_dir is None:
            # Cache the array proxies, not full volumes, in uncached mode.
            result = tuple(nib.load(str(pair[k])).dataobj for k in ('image', 'label'))
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            info = [(str(pair[k].resolve()), pair[k].stat().st_size,
                     pair[k].stat().st_mtime_ns) for k in ('image', 'label')]
            key = hashlib.sha256(json.dumps(info).encode()).hexdigest()[:20]
            paths = [self.cache_dir / f'{cid}_{key}_{kind}.npy' for kind in ('image', 'label')]
            reuse_root = os.environ.get('BALORA_READ_ONLY_CACHE_ROOT','').strip()
            reuse_paths = [Path(reuse_root)/p.name for p in paths] if reuse_root else []
            if reuse_paths and all(p.is_file() for p in reuse_paths):
                result=tuple(np.load(p,mmap_mode='r',allow_pickle=False) for p in reuse_paths)
                self._volume_cache[cid]=result
                return result
            if not all(p.is_file() for p in paths):
                needed = int(np.prod(self._shapes[cid])) * 5
                if shutil.disk_usage(self.cache_dir).free < needed + 2 * 2**30:
                    raise RuntimeError('Insufficient disk for volume cache plus 2 GiB reserve. Set BALORA_CACHE_DIR to a larger disk.')
                for kind, dest in zip(('image', 'label'), paths):
                    if dest.is_file():
                        continue
                    arr = np.asarray(nib.load(str(pair[kind])).dataobj, dtype=np.float32)
                    if kind == 'label':
                        self._validate_labels(arr, cid)
                        arr = np.round(arr).astype(np.uint8)
                    elif not np.isfinite(arr).all():
                        raise ValueError(f'{cid}: non-finite CT intensities')
                    fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix='.tmp')
                    try:
                        with os.fdopen(fd, 'wb') as f:
                            np.save(f, arr, allow_pickle=False)
                        os.replace(tmp, dest)
                    finally:
                        if os.path.exists(tmp): os.unlink(tmp)
            result = tuple(np.load(p, mmap_mode='r', allow_pickle=False) for p in paths)
        self._volume_cache[cid] = result
        return result

    def __len__(self) -> int:
        return len(self._clip_index)

    def __getitem__(self, idx: int) -> BTCVSample:
        cid, start, end = self._clip_index[idx]
        # Use nibabel's array proxy to materialize the current slice range only.
        # Loading a whole 3-D volume for every clip causes severe repeated I/O.
        im_volume, lab_volume = self._read_volume(cid)
        im = np.asarray(im_volume[:, :, start:end], dtype=np.float32)
        lab = np.asarray(lab_volume[:, :, start:end], dtype=np.float32)
        if not np.isfinite(im).all():
            raise ValueError(f'{cid}: non-finite CT intensities')
        vals = np.unique(lab)
        bad = vals[(~np.isfinite(vals)) | (np.abs(vals - np.round(vals)) > 1e-3) | (vals < 0) | (vals >= self.num_classes)]
        if bad.size:
            raise ValueError(f"{cid}: unexpected/non-integer labels {bad[:20].tolist()} with num_classes={self.num_classes}")
        low, high = self.ct_window
        imgs = np.clip(im, low, high)
        imgs = (imgs - low) / (high - low)
        lbls = np.round(lab).astype(np.int64)
        # (H,W,T) -> (T,1,H,W), and nearest-neighbour resize integer labels.
        x = torch.from_numpy(np.ascontiguousarray(imgs.transpose(2, 0, 1))).unsqueeze(1).float()
        y = torch.from_numpy(np.ascontiguousarray(lbls.transpose(2, 0, 1))).unsqueeze(1).float()
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(self.image_size, self.image_size), mode="nearest").squeeze(1).long()
        if int(y.min()) < 0 or int(y.max()) >= self.num_classes:
            raise ValueError(f"{cid}: resized labels outside [0,{self.num_classes-1}]")
        y_onehot = F.one_hot(y, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = torch.ones(x.shape[0], dtype=torch.bool)
        return BTCVSample(x, y_onehot, cid, list(range(start, end)), valid)


def collate_btcv(batch: list[BTCVSample]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty BTCV batch")
    max_t = max(s.image.shape[0] for s in batch)
    images, labels, valid_slices, all_indices = [], [], [], []
    for s in batch:
        n = s.image.shape[0]
        pad = max_t - n
        if pad:
            images.append(torch.cat([s.image, s.image[-1:].repeat(pad, 1, 1, 1)], dim=0))
            labels.append(torch.cat([s.label, s.label[-1:].repeat(pad, 1, 1, 1)], dim=0))
            valid_slices.append(torch.cat([s.valid_slices, torch.zeros(pad, dtype=torch.bool)]))
            all_indices.append(s.slice_indices + [-1] * pad)
        else:
            images.append(s.image); labels.append(s.label); valid_slices.append(s.valid_slices)
            all_indices.append(s.slice_indices)
    return {"image": torch.stack(images), "label": torch.stack(labels),
            "case_ids": [s.case_id for s in batch], "slice_indices": all_indices,
            "valid_slice_mask": torch.stack(valid_slices)}
