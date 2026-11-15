"""Discover BTCV pairs, validate NIfTI data, and save a fold manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets.splits import discover_btcv_pairs, make_fold_split, validate_btcv_pairs, write_fold_splits


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="BTCV Training root or folder containing imagesTr/labelsTr")
    p.add_argument("--output-dir", default="data/splits")
    p.add_argument("--fold", type=int, default=0); p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--scan-label-values", action="store_true", help="Full data scan; slower")
    args=p.parse_args()
    pairs=discover_btcv_pairs(args.root)
    print(f"Discovered {len(pairs)} image/label pairs below {Path(args.root).resolve()}")
    for cid, pair in pairs.items(): print(f"{cid}\timage={pair['image']}\tlabel={pair['label']}")
    report=validate_btcv_pairs(pairs, scan_label_values=args.scan_label_values)
    split=make_fold_split(list(pairs),args.n_folds,args.fold,args.seed,args.val_fraction)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/f"btcv_fold{args.fold}_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    paths=write_fold_splits(split,out)
    print(json.dumps({"case_count":len(pairs),"fold":args.fold,"split_hash":split['split_hash'],
                      "train":len(split['train']),"validation":len(split['validation']),"test":len(split['test']),
                      "files":[str(x) for x in paths]},indent=2))

if __name__ == "__main__": main()
