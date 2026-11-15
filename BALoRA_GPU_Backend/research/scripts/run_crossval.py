"""Run deterministic volume-level BTCV cross-validation and held-out evaluation.

Each fold receives a distinct train/validation/test volume manifest. This is a
long-running GPU job and should be launched only after the Kaggle smoke suite
passes. It does not fabricate results: incomplete/failed folds are recorded.
"""
from __future__ import annotations
import argparse, copy, json, math, time
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
from datasets.splits import discover_btcv_pairs, validate_btcv_pairs, make_fold_split, write_fold_splits
from datasets.btcv_dataset import BTCVDataset, collate_btcv
from training.train import run_training
from evaluation.evaluate import run_evaluation, RefinementConfig
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--sam2-root', required=True, help='Official cloned SAM2 repository root')
    ap.add_argument('--output-root', default='experiments/btcv_crossval')
    ap.add_argument('--folds', nargs='+', type=int, default=[0,1,2,3,4])
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--val-fraction', type=float, default=0.15)
    args=ap.parse_args()
    sam2_root=Path(args.sam2_root).resolve()
    if not (sam2_root/'sam2'/'build_sam.py').is_file():
        raise FileNotFoundError(f"--sam2-root is not an official SAM2 source checkout: {sam2_root}")
    import sam2
    module_path=Path(sam2.__file__).resolve()
    expected_package=(sam2_root/'sam2').resolve()
    if expected_package not in module_path.parents:
        raise RuntimeError(f"Imported SAM2 from {module_path}, not from requested source checkout {expected_package}. Resolve package shadowing before training.")

    config_path=Path(args.config).resolve()
    base=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    cases=discover_btcv_pairs(args.root)
    audit=validate_btcv_pairs(cases, num_classes=int(base['data']['num_classes']), scan_label_values=True)
    if len(cases) < args.n_folds * 2:
        raise ValueError(f'Only {len(cases)} cases found; insufficient for {args.n_folds}-fold CV with inner validation')
    seed=int(args.seed if args.seed is not None else base['experiment'].get('seed',42))
    out_root=Path(args.output_root).resolve(); out_root.mkdir(parents=True,exist_ok=True)
    (out_root/'dataset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    summaries=[]

    for fold in args.folds:
        fold_start=time.time()
        split=make_fold_split(list(cases),n_folds=args.n_folds,fold=fold,seed=seed,val_fraction=args.val_fraction)
        fold_dir=out_root/f'fold_{fold}'; fold_dir.mkdir(parents=True,exist_ok=True)
        split_paths=write_fold_splits(split,fold_dir/'splits')
        cfg=copy.deepcopy(base)
        cfg['data']['root_dir']=str(Path(args.root).resolve())
        cfg['data']['train_split']=str(split_paths[0])
        cfg['data']['val_split']=str(split_paths[1])
        cfg['data']['test_split']=str(split_paths[2])
        cfg['data']['n_folds']=args.n_folds; cfg['data']['fold']=fold; cfg['data']['split_seed']=seed
        cfg['sam2']['checkpoint']=str(Path(cfg['sam2']['checkpoint']).resolve())
        cfg['experiment']['name']=f"balora_btcv_fold{fold}"
        cfg['experiment']['seed']=seed+fold
        cfg['experiment']['output_dir']=str(fold_dir/'training')
        (fold_dir/'resolved_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        set_seed(seed+fold)
        record={'fold':fold,'split_hash':split['split_hash'],'n_train':len(split['train']),
                'n_validation':len(split['validation']),'n_test':len(split['test']),
                'status':'RUNNING','started_unix':time.time()}
        try:
            train_result=run_training(cfg)
            best=train_result.get('best_checkpoint')
            if not best or not Path(best).is_file():
                raise RuntimeError('Training finished without a finite validation checkpoint; held-out test is not run.')
            model=train_result['model']
            device=cfg['experiment'].get('device','cuda') if torch.cuda.is_available() else 'cpu'
            load_checkpoint(model,None,best,map_location=device,restore_rng=False)
            test_ds=BTCVDataset(cfg['data']['root_dir'],cfg['data']['test_split'],cfg['data']['num_classes'],
                clip_len=cfg['data'].get('clip_len',1),train=False,image_size=cfg['data']['image_size'],
                ct_window=tuple(cfg['data'].get('ct_window',[-175,250])))
            test_loader=DataLoader(test_ds,batch_size=cfg['training']['batch_size'],shuffle=False,
                                   collate_fn=collate_btcv,num_workers=cfg['training'].get('num_workers',0))
            rc=None
            if cfg.get('refinement',{}).get('enabled',False):
                r=cfg['refinement']
                rc=RefinementConfig(confidence_threshold=float(r['confidence_threshold']),
                    max_iters=int(r['max_iters']),crop_padding=int(r['crop_padding']),
                    region_min_area=int(cfg.get('confidence',{}).get('region_min_area',25)),
                    k_samples=int(cfg['bayesian']['k_infer']))
            pred_dir=fold_dir/'predictions'
            eval_result=run_evaluation(model,test_loader,device,k_infer=int(cfg['bayesian']['k_infer']),
                                       refinement_config=rc,compute_surface_distances=True,
                                       prediction_dir=pred_dir)
            (fold_dir/'heldout_evaluation.json').write_text(json.dumps(eval_result,indent=2),encoding='utf-8')
            record.update({'status':'COMPLETED','best_checkpoint':best,
                           'validation_history':train_result.get('history'),
                           'heldout_evaluation_path':str(fold_dir/'heldout_evaluation.json'),
                           'heldout_segmentation':eval_result.get('segmentation'),
                           'heldout_calibration':eval_result.get('calibration')})
            del model, test_loader, test_ds, train_result
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as exc:
            record.update({'status':'FAILED','error_type':type(exc).__name__,'error':str(exc)})
            (fold_dir/'failure.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            summaries.append(record)
            # Stop by default: a failed fold can indicate implementation/data errors.
            (out_root/'crossval_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
            raise
        record['duration_seconds']=time.time()-fold_start
        summaries.append(record)
        (out_root/'crossval_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
        print(json.dumps({'fold':fold,'status':record['status'],'seconds':record['duration_seconds'],
                          'heldout_segmentation':record.get('heldout_segmentation')},indent=2))

    if len(summaries) != len(args.folds) or any(r['status']!='COMPLETED' for r in summaries):
        raise RuntimeError('Cross-validation did not complete all requested folds')
    metric_names=sorted(set().union(*(r.get('heldout_segmentation',{}).keys() for r in summaries)))
    aggregate={}
    for name in metric_names:
        vals=[]
        for rec in summaries:
            item=rec.get('heldout_segmentation',{}).get(name)
            if isinstance(item,dict) and isinstance(item.get('mean'),(int,float)) and math.isfinite(item['mean']):
                vals.append(float(item['mean']))
        if vals:
            aggregate[name]={'mean_across_folds':sum(vals)/len(vals),'fold_values':vals,'n_folds':len(vals)}
    payload={'status':'COMPLETED','n_requested_folds':len(args.folds),'n_completed_folds':len(summaries),
             'fold_results':summaries,'unweighted_fold_mean_summary':aggregate,
             'note':'Fold-level summaries; for publication, preserve per-volume predictions and compute inferential statistics at the volume level.'}
    (out_root/'crossval_aggregate.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate_path':str(out_root/'crossval_aggregate.json'),'summary':aggregate},indent=2))

if __name__=='__main__': main()
