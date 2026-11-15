"""Same-split, same-prompt frozen SAM2 baseline; no training or selection."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import time
import zipfile

import torch
from evaluation.evaluate import _flatten_valid, _prediction_seed, _summarize_rows
from evaluation.metrics import compute_segmentation_metrics, compute_calibration_metrics
from evaluation.validation_audit import REFERENCE_BEST_SHA256, prepare_reference
from datasets.btcv_dataset import BTCV_ORGAN_NAMES
from utils.prompting import sample_foreground_points
from utils.runtime_budget import BudgetExpired, check_deadline, write_json

BUILD = 'zero_shot_sam2_comparison_v1'
AUDIT_FILES = ('validation_selection.json','validation_comparison.json','audit_session_state.json')
ORIGINAL_FILES = ('heldout_evaluation.json',)
RESUME_FILES = ('zero_shot_progress.json','zero_shot_evaluation.json','method_comparison.json',
                'method_comparison.csv','per_organ_comparison.csv','per_volume_comparison.csv',
                'baseline_session_state.json')


def read_bundle(source, required):
    """Read selected metadata from a folder or a safely named ZIP."""
    source = Path(source)
    if source.is_dir():
        return {name:json.loads((source/name).read_text()) for name in required}
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('Unsafe ZIP entry: '+name)
        prefixes = [n[:-len(required[0])] for n in names if n.endswith(required[0])
                    and all(n[:-len(required[0])]+r in names for r in required)]
        if len(prefixes) != 1:
            raise ValueError('Expected one completed result bundle: '+str(required))
        return {name:json.loads(z.read(prefixes[0]+name)) for name in required}


def locate_audit(input_root, explicit=''):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        read_bundle(p, AUDIT_FILES)
        return p
    root = Path(input_root)
    candidates = [p.parent for p in root.rglob(AUDIT_FILES[0])
                  if all((p.parent/n).is_file() for n in AUDIT_FILES)]
    if not candidates:
        for p in root.rglob('*.zip'):
            if 'validation_audit_results' in p.name.lower():
                read_bundle(p, AUDIT_FILES)
                candidates.append(p)
    if len(candidates) != 1:
        raise FileNotFoundError('Attach the completed validation audit output or ZIP. '
                                'Set AUDIT_INPUT when several outputs match.')
    return candidates[0]


def locate_original_results(input_root, explicit=''):
    """Exclude the audit's checkpoint-only reference copy from discovery."""
    required=('heldout_evaluation.json','reproducibility_manifest.json',
              'selected_checkpoint.json','session_state.json')
    if explicit:
        p=Path(explicit).expanduser().resolve()
        read_bundle(p,required)
        return p
    root=Path(input_root)
    folders=[p.parent for p in root.rglob('heldout_evaluation.json')
             if all((p.parent/name).is_file() for name in required)
             and (p.parent/'research_fold_01/best.pt').is_file()]
    if len(folders)==1:
        return folders[0]
    if len(folders)>1:
        raise ValueError('Several original test outputs match; set RESULTS_INPUT explicitly')
    archives=[]
    for p in root.rglob('*.zip'):
        if 'balora' not in p.name.lower() or 'results' not in p.name.lower():
            continue
        try:
            read_bundle(p,required)
        except (ValueError,KeyError):
            continue
        archives.append(p)
    if len(archives)!=1:
        raise FileNotFoundError('Attach the original completed Fold-1 results with heldout_evaluation.json. '
                                'Set RESULTS_INPUT explicitly when necessary.')
    return archives[0]


def prepare_comparison_context(results_source, audit_source, local_inputs):
    reference = prepare_reference(results_source, local_inputs, REFERENCE_BEST_SHA256)
    original = read_bundle(results_source, ORIGINAL_FILES)['heldout_evaluation.json']
    audit = read_bundle(audit_source, AUDIT_FILES)
    selection, comparison, usage = [audit[n] for n in AUDIT_FILES]
    if not selection.get('complete') or selection.get('selected_policy') != 'none':
        raise ValueError('This comparison requires completed validation selecting no refinement')
    if selection.get('test_evaluated') is not False or comparison.get('test_evaluated') is not False:
        raise ValueError('The policy must have been selected using validation only')
    if not comparison.get('complete') or comparison.get('checkpoint_sha256') != reference['checkpoint_sha256']:
        raise ValueError('Audit does not match the selected checkpoint')
    if comparison.get('model_fingerprint') != selection.get('model_fingerprint'):
        raise ValueError('Audit model identity mismatch')
    if set(selection['validation_cases']) != set(reference['splits']['validation']):
        raise ValueError('Validation case identities changed')
    scores = selection['validation_macro_organ_dice']
    if scores['guarded'] > scores['none']+1e-6:
        raise ValueError('Saved validation decision contradicts its declared rule')
    if (not usage.get('closed_cleanly') or usage.get('execution_status') != 'VALIDATION_AUDIT_COMPLETE'
            or usage.get('checkpoint_sha256') != reference['checkpoint_sha256']):
        raise ValueError('Completed audit usage ledger is required')
    proto = original['protocol']
    if (not proto.get('complete') or proto.get('bounded_diagnostic')
            or original['selected_checkpoint']['sha256'] != reference['checkpoint_sha256']):
        raise ValueError('Original test is incomplete or belongs to other weights')
    test = set(reference['splits']['test'])
    base = original['same_checkpoint_no_refinement']
    scored = {cid for cases in base['per_volume_segmentation'].values() for cid in cases}
    if scored != test or len(test) != 6:
        raise ValueError('Original no-refinement test coverage differs from the saved split')
    counts = {(organ,cid):record['n_slices']
              for organ,cases in base['per_volume_segmentation'].items()
              for cid,record in cases.items()}
    if sum(counts.values()) != proto['n_scored_slice_organ_predictions']:
        raise ValueError('Original slice/organ counts are inconsistent')
    prior = max(float(usage['accounted_runtime_seconds']),reference['prior_accounted_seconds'])
    if not math.isfinite(prior) or prior < 0:
        raise ValueError('Invalid cumulative usage')
    return {'reference':reference,'original_no_refinement':base,'expected_counts':counts,
            'seed':int(proto['seed']),'expected_predictions':int(proto['n_scored_slice_organ_predictions']),
            'policy_selection':selection,'audit_source':str(audit_source),
            'prior_accounted_seconds':prior}


def run_zero_shot_evaluation(model, loader, device, *, test_cases, seed,
                             progress_path, deadline=None):
    """One deterministic mask per prompt; preserve the original prompt RNG."""
    expected = set(test_cases)
    if set(getattr(loader.dataset,'case_ids',[])) != expected:
        raise ValueError('Only the saved test cases may be evaluated')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('The zero-shot backbone must be frozen')
    model.eval()
    identity = {'build':BUILD,'model':model.identity,'seed':int(seed),'test_cases':sorted(expected),
                'dataset_index':getattr(loader.dataset,'_clip_index',None),
                'surface_distances':True,'batch_size':loader.batch_size,
                'prompt_protocol':'original foreground sampler, seeded per case/slice/organ',
                'refinement':False,'posterior_sampling':False}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    progress_path = Path(progress_path)
    rows = []
    if progress_path.is_file():
        saved = json.loads(progress_path.read_text())
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Baseline progress belongs to another backbone/data/protocol')
        rows = saved['rows']
    done = {(r['case_id'],r['slice_index'],r['organ_class']) for r in rows}
    if len(done) != len(rows):
        raise ValueError('Duplicate baseline prediction records')
    started = time.time()
    complete = False
    batch_idx = -1
    def persist():
        write_json(progress_path, {'fingerprint':fingerprint,'identity':identity,'rows':rows,
                                  'complete':complete,'last_batch_index':batch_idx})
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            for batch_idx,batch in enumerate(loader):
                check_deadline(deadline,'zero-shot baseline')
                images,labels,metadata = _flatten_valid(batch,device)
                if len(metadata) != len(images):
                    raise ValueError('Baseline metadata/image count mismatch')
                needed = []
                for i,(cid,z) in enumerate(metadata):
                    if cid not in expected:
                        raise ValueError('Unexpected volume reached baseline evaluation')
                    classes = [c for c in range(1,labels.shape[1]) if bool(labels[i,c].any())]
                    if any((cid,z,c) not in done for c in classes):
                        needed.append((i,cid,z,classes))
                if not needed:
                    continue
                embeddings = model.encode_image(images)
                for i,cid,z,classes in needed:
                    emb = embeddings[i:i+1]
                    for c in classes:
                        if (cid,z,c) in done:
                            continue
                        check_deadline(deadline,'zero-shot organ')
                        # Same function, device and first RNG draw as original
                        # BALoRA evaluation; CPU/CUDA samplers are not mixed.
                        torch.manual_seed(_prediction_seed(seed,cid,z,c))
                        target = labels[i:i+1,c]
                        points = sample_foreground_points(target)
                        point_labels = torch.ones((1,1),dtype=torch.int32,device=target.device)
                        logits = model.decode_one(emb,points,point_labels)
                        if logits.shape != (1,1,*target.shape[-2:]) or not torch.isfinite(logits).all():
                            raise FloatingPointError('Invalid native SAM2 prediction')
                        prob = logits[0,0].float().sigmoid().cpu()
                        truth = target[0].cpu()
                        seg = compute_segmentation_metrics(prob>=0.5,truth,True).to_dict()
                        cal = compute_calibration_metrics(prob,truth).to_dict()
                        if not all(math.isfinite(v) for v in cal.values()):
                            raise FloatingPointError('Non-finite baseline calibration')
                        rows.append({'case_id':cid,'slice_index':int(z),'organ_class':c,
                                     'organ':BTCV_ORGAN_NAMES[c],'seg':seg,'cal':cal,
                                     'prompt_xy':points[0,0].cpu().tolist(),
                                     'empty_prediction':not bool((prob>=0.5).any())})
                        done.add((cid,z,c))
                del embeddings
                if (batch_idx+1)%20 == 0:
                    persist()
                    print(f'zero-shot batches={batch_idx+1} predictions={len(rows)} elapsed_min={(time.time()-started)/60:.1f}',flush=True)
            complete = True
    except BudgetExpired:
        complete = False
    finally:
        persist()
    if complete and not rows:
        raise RuntimeError('No baseline predictions were scored')
    seg,org,vol,undef = _summarize_rows(rows,'seg')
    cal,org_cal,_,undef_cal = _summarize_rows(rows,'cal')
    return {'method':'official SAM2 zero-shot; no adapters','complete':complete,
            'protocol':{'complete':complete,'stop_reason':'COMPLETED' if complete else 'BUDGET_PAUSED',
                        'test_cases':sorted(expected),'seed':seed,'refinement':False,
                        'deterministic_prediction_per_prompt':True,'posterior_samples':None,
                        'epistemic_uncertainty_estimated':False,'training_updates':0,
                        'aggregation_unit':'volume-organ pair; slice metrics averaged within each pair',
                        'surface_distance_units':'pixels at resized 2D resolution, not 3D millimetres',
                        'prompt_policy':'one ground-truth-positive point per present organ slice',
                        'n_scored_slice_organ_predictions':len(rows)},
            'segmentation':seg,'calibration':cal,'per_organ_segmentation':org,
            'per_organ_calibration':org_cal,'per_volume_segmentation':vol,
            'undefined_slice_metrics':{'segmentation':undef,'calibration':undef_cal},
            'empty_prediction_count':sum(r['empty_prediction'] for r in rows),
            'elapsed_seconds_this_session':time.time()-started,'fingerprint':fingerprint,
            'research_result':not model.test_backbone_injected}


def export_comparison(context, baseline, output_root):
    if not baseline['complete']:
        raise ValueError('A partial baseline cannot be exported as a completed comparison')
    actual = {(organ,cid):r['n_slices'] for organ,cases in baseline['per_volume_segmentation'].items()
              for cid,r in cases.items()}
    if actual != context['expected_counts']:
        raise ValueError('Methods do not cover identical volume/organ slices')
    original = context['original_no_refinement']
    root = Path(output_root);root.mkdir(parents=True,exist_ok=True)
    rows = []
    for name,report in [('SAM2 zero-shot',baseline),('BALoRA, no refinement',original)]:
        rows.append({'method':name,'dice':report['segmentation']['dice']['mean'],
                     'dice_sd_volume_organ_pairs':report['segmentation']['dice']['std'],
                     'n_volume_organ_pairs':report['segmentation']['dice']['n'],
                     'iou':report['segmentation']['iou']['mean'],
                     'hd95_pixels':report['segmentation'].get('hd95',{}).get('mean'),
                     'assd_pixels':report['segmentation'].get('assd',{}).get('mean'),
                     'ece':report['calibration']['ece']['mean'],
                     'brier':report['calibration']['brier']['mean'],
                     'nll':report['calibration']['nll']['mean']})
    organ_rows=[];volume_rows=[]
    for organ in sorted(baseline['per_organ_segmentation']):
        a=baseline['per_organ_segmentation'][organ]['dice']['mean']
        b=original['per_organ_segmentation'][organ]['dice']['mean']
        organ_rows.append({'organ':organ,'sam2_zero_shot_dice':a,'balora_no_refinement_dice':b,'balora_minus_sam2':b-a})
        for cid in sorted(baseline['per_volume_segmentation'][organ]):
            a=baseline['per_volume_segmentation'][organ][cid]['metrics']['dice']
            b=original['per_volume_segmentation'][organ][cid]['metrics']['dice']
            volume_rows.append({'case_id':cid,'organ':organ,'sam2_zero_shot_dice':a,
                                'balora_no_refinement_dice':b,'balora_minus_sam2':b-a})
    for filename,data in [('method_comparison.csv',rows),('per_organ_comparison.csv',organ_rows),
                           ('per_volume_comparison.csv',volume_rows)]:
        with (root/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    report={'complete':True,'methods':rows,'checkpoint_sha256':context['reference']['checkpoint_sha256'],
            'validation_selected_policy':'none','training_updates_this_stage':0,
            'test_cases':context['reference']['splits']['test'],'prompt_seed':context['seed'],
            'identical_case_organ_slice_coverage_verified':True,
            'n_predictions_per_method':context['expected_predictions'],
            'note':'Fixed-method exploratory comparison after original Fold-1 test inspection. '
                   'Original BALoRA results are reused; no checkpoint/policy is chosen from these scores. '
                   'This is a frozen zero-shot reference, not a trained fixed-rank LoRA ablation.'}
    write_json(root/'method_comparison.json',report)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        x=np.arange(len(organ_rows));fig,ax=plt.subplots(figsize=(12,5))
        ax.bar(x-.2,[r['sam2_zero_shot_dice'] for r in organ_rows],.4,label='SAM2 zero-shot')
        ax.bar(x+.2,[r['balora_no_refinement_dice'] for r in organ_rows],.4,label='BALoRA, no refinement')
        ax.set_xticks(x,[r['organ'].replace('_',' ') for r in organ_rows],rotation=55,ha='right')
        ax.set_ylabel('Dice');ax.set_ylim(0,1);ax.legend();fig.tight_layout()
        fig.savefig(root/'per_organ_dice.png',dpi=180);plt.close(fig)
    except ImportError:
        pass
    return report
