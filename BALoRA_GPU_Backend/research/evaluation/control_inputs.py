"""Freeze control design and cumulative accounting from completed references."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from evaluation.validation_audit import prepare_reference, validate_saved_split
from evaluation.zero_shot import read_bundle
from utils.runtime_budget import write_json

BUILD='deterministic_lora_control_v1'
ZERO_FILES=('baseline_session_state.json','method_comparison.json','zero_shot_evaluation.json',
            'comparison_reference.json','frozen_policy.json','baseline_config.json')


def locate_zero_shot(input_root,explicit=''):
    if explicit:
        path=Path(explicit).expanduser().resolve();read_bundle(path,ZERO_FILES);return path
    root=Path(input_root)
    paths=[p.parent for p in root.rglob(ZERO_FILES[0]) if all((p.parent/n).is_file() for n in ZERO_FILES)]
    if not paths:
        for p in root.rglob('*.zip'):
            if 'zeroshot_comparison_results' in p.name.lower():
                read_bundle(p,ZERO_FILES);paths.append(p)
    if len(paths)!=1:
        raise FileNotFoundError('Attach the completed BALoRA_ZeroShot_Comparison_Results.zip or its extracted output; set ZERO_SHOT_INPUT if several match')
    return paths[0]


def prepare_control_context(original_source,zero_source,local_inputs):
    ref=prepare_reference(original_source,local_inputs)
    zero=read_bundle(zero_source,ZERO_FILES)
    state,table,baseline,comparison,policy,zero_cfg=[zero[n] for n in ZERO_FILES]
    if not state.get('closed_cleanly') or state.get('execution_status')!='ZERO_SHOT_COMPARISON_COMPLETE':
        raise ValueError('A clean completed zero-shot ledger is required')
    if not table.get('complete') or not baseline.get('complete') or not baseline['protocol'].get('complete'):
        raise ValueError('Zero-shot comparison is incomplete')
    if any(x!=ref['checkpoint_sha256'] for x in [state.get('checkpoint_sha256'),table.get('checkpoint_sha256'),comparison.get('selected_checkpoint_sha256')]):
        raise ValueError('Reference checkpoint identity mismatch')
    original_report=read_bundle(original_source,('heldout_evaluation.json',))['heldout_evaluation.json']
    if not original_report['protocol'].get('complete') or original_report.get('same_checkpoint_no_refinement')!=comparison['original_balora_no_refinement']:
        raise ValueError('Copied BALoRA results differ from the original completed test report')
    for key in ('model_cfg','checkpoint_sha256','resolved_commit'):
        if zero_cfg['sam2'][key]!=ref['configuration']['sam2'][key]:
            raise ValueError('Frozen zero-shot backbone differs from the original reference')
    for key in ('image_size','num_classes','ct_window'):
        if zero_cfg['data'][key]!=ref['configuration']['data'][key]:
            raise ValueError('Frozen zero-shot preprocessing differs from the original reference')
    if (not policy.get('complete') or policy.get('selected_policy')!='none'
            or policy.get('test_evaluated') is not False
            or set(policy['validation_cases'])!=set(ref['splits']['validation'])
            or policy['validation_macro_organ_dice']['guarded']>policy['validation_macro_organ_dice']['none']+1e-6):
        raise ValueError('Completed validation must retain the no-refinement policy')
    expected={cid for cid in ref['splits']['test']}
    if set(table['test_cases'])!=expected or set(baseline['protocol']['test_cases'])!=expected:
        raise ValueError('Zero-shot test cases differ from the original split')
    seed=int(ref['configuration']['experiment']['seed'])+700001
    if table['prompt_seed']!=seed or comparison['test_seed']!=seed or baseline['protocol']['seed']!=seed:
        raise ValueError('Original test prompt seed changed')
    counts={(r['organ'],r['case_id']):int(r['n_slices']) for r in comparison['expected_slice_counts']}
    for report in [baseline,comparison['original_balora_no_refinement']]:
        actual={(o,c):int(r['n_slices']) for o,vols in report['per_volume_segmentation'].items() for c,r in vols.items()}
        if actual!=counts:raise ValueError('Reference comparison slice counts differ')
    if sum(counts.values())!=table['n_predictions_per_method'] or len(counts)!=78:
        raise ValueError('Incomplete reference coverage')
    prior=max(float(state['accounted_runtime_seconds']),float(ref['prior_accounted_seconds']))
    if not math.isfinite(prior) or prior<0:raise ValueError('Invalid prior usage')
    return {'reference':ref,'zero_shot_source':str(zero_source),'zero_shot':baseline,
            'original_balora':comparison['original_balora_no_refinement'],
            'policy':policy,'seed':seed,'expected_counts':counts,
            'prior_accounted_seconds':prior}


def protocol_identity(cfg):
    return {'build':BUILD,'sam2':{k:cfg['sam2'][k] for k in ('model_cfg','checkpoint_sha256','resolved_commit')},
        'data':{k:cfg['data'][k] for k in ('image_size','num_classes','ct_window','foreground_only','clip_len')},
        'experiment':{k:cfg['experiment'][k] for k in ('split_case_ids','split_hash','seed')},
        'lora':cfg['lora'],'loss':cfg['loss'],
        'training':{k:cfg['training'][k] for k in ('num_epochs','min_epochs','early_stopping_patience',
            'batch_size','lr','weight_decay','grad_clip_norm','amp')},
        'selection':'fixed validation prompts; validation Dice only; strict improvement',
        'no_refinement':True}


def protocol_digest(cfg):
    return hashlib.sha256(json.dumps(protocol_identity(cfg),sort_keys=True).encode()).hexdigest()


def make_control_config(context,btcv_root,output_root,sam2_checkpoint):
    original=context['reference']['configuration'];root=Path(output_root)
    cfg={k:copy.deepcopy(original[k]) for k in ('sam2','data','training','loss')}
    old_l=original['lora'];rank=max(old_l['rank_bank'])
    cfg['lora']={k:copy.deepcopy(old_l[k]) for k in ('encoder_target_modules','target_modules','alpha','dropout')}
    cfg['lora'].update(encoder_rank=rank,decoder_rank=rank)
    cfg['loss'].update(lambda_kl=0.,lambda_rank=0.)
    cfg['experiment']={k:copy.deepcopy(original['experiment'][k]) for k in ('seed','split_hash','split_case_ids')}
    cfg['experiment'].update(mode=BUILD,device='cuda:0',output_dir=str(root))
    cfg['training'].pop('continuation_lr',None)
    cfg['training'].update(num_epochs=20,min_epochs=8,early_stopping_patience=6,
                           batch_size=1,num_workers=0,checkpoint_seconds=300)
    cfg['sam2']['checkpoint']=str(sam2_checkpoint)
    cfg['data'].update(root_dir=str(btcv_root),clip_len=1,foreground_only=True,
                       cache_dir=str(root/'raw_volume_cache'))
    for short,split in [('train','train'),('val','validation'),('test','test')]:
        path=root/'splits'/(split+'.json');write_json(path,{'cases':context['reference']['splits'][split]})
        cfg['data'][short+'_split']=str(path)
    validate_saved_split(cfg)
    cfg['control']={'build':BUILD,'protocol_sha256':protocol_digest(cfg),
        'rank_reason':'Both ranks fixed at the original maximum (16); not chosen from test scores',
        'initialization':'Fresh official backbone and new deterministic A/B adapters; never BALoRA weights',
        'scope':'Combined point-estimate/fixed-rank control; not separate Bayesian-only and rank-only ablations',
        'test_seed':context['seed'],'policy':'none',
        'reference_checkpoint_sha256':context['reference']['checkpoint_sha256']}
    write_json(root/'control_config.json',cfg)
    write_json(root/'frozen_protocol.json',{'identity':protocol_identity(cfg),**cfg['control']})
    return cfg
