"""Volume-aware evaluation for prompt-conditioned single-organ 2D segmentation.

Metrics are calculated on slices where that organ is present because the
current protocol uses a ground-truth foreground point prompt. This evaluation
is therefore an oracle-prompt segmentation experiment, not fully automatic
multi-organ inference. The report records this explicitly and averages slice
metrics within each volume before the across-volume summary.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse, json
from pathlib import Path
from typing import Any
import torch
import yaml
from torch.utils.data import DataLoader
from datasets.btcv_dataset import BTCVDataset, BTCV_ORGAN_NAMES, collate_btcv
from losses.losses import LossWeights
from models.confidence import estimate_uncertainty
from models.refinement import RefinementConfig, refine
from models.refinement_guarded import GuardedRefinementConfig, refine_guarded
from models.sam2_wrapper import Sam2BayesianLoRAConfig, Sam2BayesianLoRAModel
from training.train import build_model
from utils.checkpoint import load_checkpoint
from utils.prompting import sample_foreground_points
from evaluation.metrics import compute_segmentation_metrics, compute_calibration_metrics


def _finite_summary(values: list[float]) -> dict[str, float | int | None]:
    import math, statistics
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def _flatten_valid(batch: dict[str, Any], device: str):
    x = batch["image"].to(device, non_blocking=True)
    y = batch["label"].to(device, non_blocking=True)
    b,t = x.shape[:2]
    valid = batch.get("valid_slice_mask", torch.ones((b,t), dtype=torch.bool)).reshape(-1).to(device)
    x = x.reshape(b*t,*x.shape[2:])[valid]
    y = y.reshape(b*t,*y.shape[2:])[valid]
    meta = []
    for i,cid in enumerate(batch["case_ids"]):
        for z in batch["slice_indices"][i]:
            if int(z) >= 0: meta.append((cid, int(z)))
    return x,y,meta


def _prediction_seed(seed, cid, z, organ):
    import hashlib
    key = f'{seed}:{cid}:{z}:{organ}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little') % (2**31-1)


def _model_fingerprint(model):
    import hashlib
    h = hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode()); h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    h.update(str(getattr(model,'_run_config',{}).get('sam2',{})).encode())
    return h.hexdigest()


def _summarize_rows(rows, prefix):
    import math
    by_organ = defaultdict(lambda:defaultdict(list))
    for row in rows:
        by_organ[row['organ']][row['case_id']].append(row[prefix])
    pooled = defaultdict(list); per_organ = {}; per_volume = {}; undefined = defaultdict(int)
    for organ,volumes in by_organ.items():
        vals_by_metric = defaultdict(list); per_volume[organ] = {}
        for cid,metrics in volumes.items():
            means = {}
            for name in metrics[0]:
                values = [r.get(name) for r in metrics]
                valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
                undefined[name] += len(values)-len(valid)
                means[name] = sum(valid)/len(valid) if valid else None
                if valid:
                    vals_by_metric[name].append(means[name]); pooled[name].append(means[name])
            per_volume[organ][cid] = {'n_slices':len(metrics),'metrics':means}
        per_organ[organ] = {name:_finite_summary(vals) for name,vals in vals_by_metric.items()}
    return {k:_finite_summary(v) for k,v in pooled.items()},per_organ,per_volume,dict(undefined)


def run_evaluation(model: Sam2BayesianLoRAModel, loader: DataLoader, device: str,
                   k_infer: int = 4, refinement_config: RefinementConfig | None = None,
                   compute_surface_distances: bool = True, max_batches: int | None = None,
                   prediction_dir: str | Path | None = None, seed: int = 42,
                   deadline: float | None = None, progress_path: str | Path | None = None,
                   include_unrefined: bool = False, progress_every: int = 50) -> dict[str, Any]:
    """Evaluate all requested organ-positive slices with deterministic prompts.

    Never use inference_mode(): rank attribution explicitly enables autograd.
    Evaluation restores the training RNG state. Progress can resume with the
    SAME model/protocol; partial reports never claim a completed held-out test.
    """
    import hashlib,time,math
    from dataclasses import asdict
    from utils.runtime_budget import BudgetExpired,check_deadline,write_json
    if k_infer < 2: raise ValueError('k_infer >=2 is required')
    model.eval(); started=time.time()
    prediction_root = Path(prediction_dir) if prediction_dir is not None else None
    if prediction_root: prediction_root.mkdir(parents=True,exist_ok=True)
    identity = {'evaluation_revision':'validation_refinement_audit_v1',
        'model':_model_fingerprint(model),'k_infer':int(k_infer),'seed':int(seed),
        'refinement':asdict(refinement_config) if refinement_config is not None else None,
        'surface':bool(compute_surface_distances),'include_unrefined':bool(include_unrefined),
        'max_batches':max_batches,'batch_size':loader.batch_size,
        'dataset_index':getattr(loader.dataset,'_clip_index',None),
        'dataset_cases':getattr(loader.dataset,'case_ids',None),
        'prediction_dir':str(prediction_root) if prediction_root else None}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
    rows=[]
    if progress_path and Path(progress_path).is_file():
        saved=json.loads(Path(progress_path).read_text())
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('Evaluation progress belongs to different weights/data/protocol; choose a new progress path')
        rows=saved['rows']
    done={(r['case_id'],r['slice_index'],r['organ_class']) for r in rows}
    complete=False; processed_batches=0; seen_images=set(); batch_idx=-1
    # torch.manual_seed also seeds every visible CUDA device. Preserve every
    # device RNG even when a CPU regression test runs in a GPU notebook.
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    def persist():
        if progress_path:
            write_json(progress_path,{'fingerprint':fingerprint,'identity':identity,'rows':rows,
                                     'complete':complete,'last_batch_index':batch_idx})
    try:
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            for batch_idx,batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches: break
                check_deadline(deadline,'evaluation')
                images,labels,metadata=_flatten_valid(batch,device)
                if images.shape[0] == 0: continue
                if len(metadata) != len(images): raise ValueError('Evaluation image/metadata count mismatch')
                seen_images.update(metadata); processed_batches+=1
                # Do not encode all-background or already completed images.
                needed=[]
                for i,(cid,z) in enumerate(metadata):
                    classes=[c for c in range(1,labels.shape[1]) if bool(labels[i,c].any())]
                    if any((cid,z,c) not in done for c in classes): needed.append((i,cid,z,classes))
                if not needed: continue
                emb=model.encode_image(images)
                for i,cid,z,classes in needed:
                    emb_i=emb[i:i+1]
                    for class_id in classes:
                        if (cid,z,class_id) in done: continue
                        check_deadline(deadline,'evaluation organ')
                        record_seed=_prediction_seed(seed,cid,z,class_id)
                        torch.manual_seed(record_seed)
                        target=labels[i:i+1,class_id]
                        points=sample_foreground_points(target)
                        point_labels=torch.ones((1,1),dtype=torch.int32,device=target.device)
                        logits,_probe,ranks=model.decode_with_adaptive_rank(emb_i,points,point_labels,k_samples=k_infer)
                        uq=estimate_uncertainty(logits.squeeze(2))
                        base_prob=uq.mean_prob[0].detach()
                        prob=base_prob; epi=uq.epistemic[0].detach(); trace=[]
                        if refinement_config is not None and refinement_config.max_iters > 0:
                            if isinstance(refinement_config, GuardedRefinementConfig):
                                prob,epi,trace=refine_guarded(model,emb_i,prob,epi,points,
                                                            refinement_config,deadline=deadline)
                            else:
                                attr=model.make_region_attribution_fn(emb_i,point_labels,k_samples=2)
                                prob,epi,trace=refine(model,emb_i,prob,epi,attr,refinement_config)
                        prob=prob.detach().float().cpu(); target_cpu=target[0].detach().cpu()
                        if not torch.isfinite(prob).all(): raise FloatingPointError('Non-finite evaluation probabilities')
                        seg=compute_segmentation_metrics(prob>=0.5,target_cpu,compute_surface_distances).to_dict()
                        cal=compute_calibration_metrics(prob,target_cpu).to_dict()
                        if not all(math.isfinite(v) for v in cal.values()):
                            raise FloatingPointError('Non-finite calibration metric')
                        row={'case_id':cid,'slice_index':int(z),'organ_class':class_id,
                            'organ':BTCV_ORGAN_NAMES[class_id],'seg':seg,'cal':cal,
                            'refinement_iters':sum(bool(getattr(t,'accepted',True)) for t in trace),
                            'refinement_attempts':len(trace),
                            'refinement_rejections':dict(Counter(
                                t.reason for t in trace if not getattr(t,'accepted',True))),
                            'ranks':ranks,
                            'empty_prediction':not bool((prob>=0.5).any())}
                        if include_unrefined:
                            bp=base_prob.cpu()
                            row['base_seg']=compute_segmentation_metrics(bp>=0.5,target_cpu,compute_surface_distances).to_dict()
                            row['base_cal']=compute_calibration_metrics(bp,target_cpu).to_dict()
                        if prediction_root:
                            import numpy as np
                            dest=prediction_root/f'{cid}_slice{z:04d}_{BTCV_ORGAN_NAMES[class_id]}.npz'
                            np.savez_compressed(dest,case_id=np.asarray(cid),slice_index=np.asarray(z,dtype=np.int32),
                                organ_class=np.asarray(class_id,dtype=np.int16),prediction=(prob>=0.5).numpy().astype(np.uint8),
                                ground_truth=target_cpu.numpy().astype(np.uint8),mean_probability=prob.numpy().astype(np.float16),
                                epistemic_uncertainty=epi.detach().cpu().numpy().astype(np.float16))
                        rows.append(row); done.add((cid,z,class_id))
                        # Release attribution graphs and all per-prediction tensors.
                        for module in model.modules():
                            if hasattr(module,'last_adapter_output'): module.last_adapter_output=None
                        del logits,_probe,uq,prob,epi,target,points,base_prob
                del emb
                if (batch_idx+1)%max(1,progress_every)==0:
                    persist()
                    print(f'evaluation batches={batch_idx+1} predictions={len(rows)} elapsed_min={(time.time()-started)/60:.1f}',flush=True)
            complete=True
    except BudgetExpired:
        complete=False
    finally:
        persist()
        for module in model.modules():
            if hasattr(module,'last_adapter_output'): module.last_adapter_output=None
    if processed_batches==0 and not rows and complete:
        raise RuntimeError('Evaluation processed no valid batches')
    seg,org_seg,vol_seg,undef_seg=_summarize_rows(rows,'seg')
    cal,org_cal,vol_cal,undef_cal=_summarize_rows(rows,'cal')
    ranks=defaultdict(list)
    for row in rows:
        for name,rank in row['ranks'].items(): ranks[name].append(rank)
    organ_means=[v['dice']['mean'] for v in org_seg.values() if v.get('dice',{}).get('mean') is not None]
    result={'protocol':{'task':'2D prompt-conditioned binary organ segmentation',
        'prompt_policy':'one positive point sampled from ground truth; fixed seed by case/slice/organ',
        'aggregation_unit':'volume-organ pair; slice scores averaged within each pair',
        'absent_organ_slices':'not scored; all-background encoding skipped',
        'processed_batches':processed_batches,'k_infer':int(k_infer),'seed':seed,
        'complete':complete,'stop_reason':'COMPLETED' if complete else 'TIME_BUDGET_PAUSED',
        'bounded_diagnostic':max_batches is not None,'refinement_enabled':bool(refinement_config and refinement_config.max_iters>0),
        'surface_distance_units':'pixels at resized 2D resolution; not 3D millimetres',
        'n_scored_volume_organ_pairs':sum(len(v) for v in vol_seg.values()),
        'n_scored_slice_organ_predictions':len(rows),'prediction_export_dir':str(prediction_root) if prediction_root else None},
        'segmentation':seg,'calibration':cal,'per_organ_segmentation':org_seg,'per_organ_calibration':org_cal,
        'per_volume_segmentation':vol_seg,'per_volume_calibration':vol_cal,
        'macro_organ_dice':sum(organ_means)/len(organ_means) if organ_means else None,
        'undefined_slice_metrics':{'segmentation':undef_seg,'calibration':undef_cal},
        'empty_prediction_count':sum(r['empty_prediction'] for r in rows),
        'rank_histogram':{n:{'mean':sum(v)/len(v),'counts':{str(r):v.count(r) for r in sorted(set(v))}} for n,v in ranks.items()},
        'refinement':{'mean_iterations':sum(r['refinement_iters'] for r in rows)/len(rows) if rows else 0.0,
                      'n_refined_predictions':sum(r['refinement_iters']>0 for r in rows),
                      'n_predictions':len(rows),
                      'attempted_updates':sum(r.get('refinement_attempts',r['refinement_iters']) for r in rows),
                      'accepted_updates':sum(r['refinement_iters'] for r in rows),
                      'rejection_counts':dict(sum((Counter(row.get('refinement_rejections',{})) for row in rows), Counter()))},
        'elapsed_seconds':time.time()-started,'evaluation_fingerprint':fingerprint}
    if include_unrefined:
        bs,bo,bv,bu=_summarize_rows(rows,'base_seg')
        bc,_,_,_=_summarize_rows(rows,'base_cal')
        result['same_checkpoint_no_refinement']={'segmentation':bs,'calibration':bc,
            'per_organ_segmentation':bo,'per_volume_segmentation':bv,
            'note':'Inference-only comparison using the same trained checkpoint, prompts and initial posterior samples; not a separately trained ablation.'}
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",default="evaluation_results.json"); parser.add_argument("--refine",action="store_true")
    parser.add_argument("--prediction-dir",default=None,help="Optional directory for per-organ, per-slice NPZ predictions")
    args=parser.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device=cfg["experiment"].get("device","cuda") if torch.cuda.is_available() else "cpu"
    model=build_model(cfg,device); load_checkpoint(model,None,args.checkpoint,map_location=device,restore_rng=False)
    ds=BTCVDataset(cfg["data"]["root_dir"],cfg["data"]["test_split"],cfg["data"]["num_classes"],
                   clip_len=cfg["data"].get("clip_len",1),train=False,image_size=cfg["data"]["image_size"])
    loader=DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=False,collate_fn=collate_btcv,num_workers=0)
    rc=None
    if args.refine:
        r=cfg["refinement"]; rc=RefinementConfig(confidence_threshold=r["confidence_threshold"],max_iters=r["max_iters"],
             crop_padding=r["crop_padding"],region_min_area=cfg["confidence"]["region_min_area"],k_samples=cfg["bayesian"]["k_infer"])
    result=run_evaluation(model,loader,device,cfg["bayesian"]["k_infer"],rc,prediction_dir=args.prediction_dir)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"output":args.output,"segmentation":result["segmentation"],"calibration":result["calibration"]},indent=2))
if __name__ == "__main__": main()
