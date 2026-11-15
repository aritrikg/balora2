from pathlib import Path
import pytest
from datasets.splits import discover_btcv_pairs, make_fold_split, split_manifest_hash


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"fake NIfTI for path discovery only")


def test_nested_synapse_layout_pairs_by_case_id(tmp_path):
    for i in (1, 2):
        touch(tmp_path / "img" / f"img{i:04d}.nii" / f"tmp-random{i}.nii")
        touch(tmp_path / "label" / f"label{i:04d}.nii" / f"tmp-label{i}.nii")
    pairs = discover_btcv_pairs(tmp_path)
    assert list(pairs) == ["img0001", "img0002"]
    assert pairs["img0001"]["image"].name.startswith("tmp-")


def test_unpaired_cases_fail_instead_of_silent_drop(tmp_path):
    touch(tmp_path / "img" / "img0001.nii")
    with pytest.raises(ValueError, match="pairing incomplete"):
        discover_btcv_pairs(tmp_path)


def test_volume_splits_are_disjoint_reproducible_and_cover_cases():
    ids = [f"img{i:04d}" for i in range(30)]
    a = make_fold_split(ids, n_folds=5, fold=2, seed=19)
    b = make_fold_split(ids, n_folds=5, fold=2, seed=19)
    assert a == b
    assert not (set(a["train"]) & set(a["validation"]))
    assert not (set(a["train"]) & set(a["test"]))
    assert not (set(a["validation"]) & set(a["test"]))
    assert set(a["train"] + a["validation"] + a["test"]) == set(ids)
    assert a["split_hash"] == split_manifest_hash({"train":a["train"],"validation":a["validation"],"test":a["test"]})
