"""Resumable, same-prompt evaluation of a trained deterministic LoRA control."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from datasets.btcv_dataset import BTCV_ORGAN_NAMES
from evaluation.evaluate import _flatten_valid, _prediction_seed, _summarize_rows
from evaluation.point_metrics import compute_point_metrics,REVISION as METRIC_REVISION
from utils.prompting import sample_foreground_points
from utils.runtime_budget import BudgetExpired, check_deadline, write_json


def adapter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate_point_model(model,loader,device,*,allowed_cases,seed,progress_path,
                         surface=False,deadline=None,split='validation'):
    expected=set(allowed_cases)
    if set(loader.dataset.case_ids)!=expected or not expected:
        raise ValueError('Evaluation cases differ from the declared split')
    if split not in ('validation','test','profiling_validation'):
        raise ValueError('Explicit validation or test split is required')
    model.assert_control();model.eval()
    digest=adapter_digest(model)
    identity={'revision':'deterministic_point_evaluation_v1_1','metric_revision':METRIC_REVISION,'model':model.identity,
              'adapters_sha256':digest,'seed':int(seed),'cases':sorted(expected),
              'dataset_index':loader.dataset._clip_index,'batch_size':loader.batch_size,
              'surface_distances':bool(surface),'split':split,'refinement':False}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=Path(progress_path);rows=[]
    if path.is_file():
        saved=json.loads(path.read_text())
        if saved['fingerprint']!=fingerprint:
            raise ValueError('Control evaluation progress belongs to another model/data/protocol')
        rows=saved['rows']
    done={(r['case_id'],r['slice_index'],r['organ_class']) for r in rows}
    if len(done)!=len(rows) or any(r['case_id'] not in expected for r in rows):
        raise ValueError('Duplicate or out-of-split saved evaluation rows')
    complete=False;batch_index=-1;started=time.time()
    def persist():
        write_json(path,{'identity':identity,'fingerprint':fingerprint,'rows':rows,
                         'complete':complete,'last_batch_index':batch_index})
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices),torch.inference_mode():
            for batch_index,batch in enumerate(loader):
                check_deadline(deadline,'control '+split)
                images,labels,metadata=_flatten_valid(batch,device)
                if len(metadata)!=len(images):raise ValueError('Evaluation metadata mismatch')
                needed=[]
                for i,(cid,z) in enumerate(metadata):
                    if cid not in expected:raise ValueError('Out-of-split evaluation volume')
                    classes=[c for c in range(1,labels.shape[1]) if bool(labels[i,c].any())]
                    if any((cid,z,c) not in done for c in classes):needed.append((i,cid,z,classes))
                if not needed:continue
                embedding=model.encode_image(images)
                for i,cid,z,classes in needed:
                    for c in classes:
                        if (cid,z,c) in done:continue
                        check_deadline(deadline,'control evaluation organ')
                        torch.manual_seed(_prediction_seed(seed,cid,z,c))
                        target=labels[i:i+1,c]
                        points=sample_foreground_points(target)
                        point_labels=torch.ones(points.shape[:2],device=target.device,dtype=torch.int32)
                        logits=model.decode_one(embedding[i:i+1],points,point_labels)
                        if logits.shape!=(1,1,*target.shape[-2:]) or not torch.isfinite(logits).all():
                            raise FloatingPointError('Invalid control mask logits')
                        prob=logits[0,0].float().sigmoid().cpu();truth=target[0].cpu()
                        seg_result,cal_result,empty=compute_point_metrics(prob,truth,surface)
                        seg=seg_result.to_dict();cal=cal_result.to_dict()
                        if not all(math.isfinite(v) for v in cal.values()):
                            raise FloatingPointError('Non-finite control calibration')
                        rows.append({'case_id':cid,'slice_index':int(z),'organ_class':c,
                            'organ':BTCV_ORGAN_NAMES[c],'seg':seg,'cal':cal,
                            'prompt_xy':points[0,0].cpu().tolist(),
                            'empty_prediction':empty})
                        done.add((cid,z,c))
                del embedding
                if (batch_index+1)%20==0:
                    persist()
                    print(f'control {split}: batches={batch_index+1} predictions={len(rows)} elapsed_min={(time.time()-started)/60:.1f}',flush=True)
            complete=True
    except BudgetExpired:
        complete=False
    finally:
        persist()
    if adapter_digest(model)!=digest:raise RuntimeError('Evaluation mutated the control adapters')
    if complete and not rows:raise ValueError('No control predictions were scored')
    seg,org,vol,undef=_summarize_rows(rows,'seg')
    cal,org_cal,vol_cal,undef_cal=_summarize_rows(rows,'cal')
    return {'complete':complete,'method':'Deterministic fixed-rank LoRA',
        'protocol':{'complete':complete,'stop_reason':'COMPLETED' if complete else 'BUDGET_PAUSED',
            'split':split,'cases':sorted(expected),'seed':int(seed),'refinement':False,
            'posterior_sampling':False,'epistemic_uncertainty_estimated':False,
            'prompt_policy':'one ground-truth-positive point per present organ slice',
            'aggregation_unit':'volume-organ pair; slice metrics averaged within each pair',
            'surface_distance_units':'pixels at resized 2D resolution, not 3D millimetres',
            'n_scored_slice_organ_predictions':len(rows),'n_scored_volume_organ_pairs':sum(len(v) for v in vol.values())},
        'segmentation':seg,'calibration':cal,'per_organ_segmentation':org,
        'per_organ_calibration':org_cal,'per_volume_segmentation':vol,'per_volume_calibration':vol_cal,
        'undefined_slice_metrics':{'segmentation':undef,'calibration':undef_cal},
        'empty_prediction_count':sum(r['empty_prediction'] for r in rows),
        'fingerprint':fingerprint,'adapters_sha256':digest,
        'elapsed_seconds_this_session':time.time()-started,
        'research_result':not model.test_backbone_injected,
        'execution_revision':METRIC_REVISION}
