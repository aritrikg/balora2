"""Budgeted orchestration for the frozen SAM2 reference evaluation."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import torch
from torch.utils.data import DataLoader
from datasets.btcv_dataset import BTCVDataset, collate_btcv
from evaluation.validation_audit import sha256_file
from evaluation.zero_shot import (BUILD, RESUME_FILES, export_comparison,
                                  run_zero_shot_evaluation)
from models.frozen_sam2 import FrozenSAM2
from utils.runtime_budget import write_json
from utils.seed import set_seed


def recover_baseline(input_root, output_root, context, explicit_root=''):
    output_root = Path(output_root)
    possible = [Path(explicit_root)/'baseline_session_state.json'] if explicit_root else \
               [output_root/'baseline_session_state.json', *Path(input_root).rglob('baseline_session_state.json')]
    matches = []
    for path in possible:
        if not path.is_file():
            continue
        state = json.loads(path.read_text())
        if state.get('build') == BUILD and state.get('checkpoint_sha256') == context['reference']['checkpoint_sha256']:
            matches.append((path,state))
    if len(matches)>1:
        raise ValueError('Several baseline outputs match; set BASELINE_RESUME_ROOT explicitly')
    prior = float(context['prior_accounted_seconds'])
    if matches:
        path,state = matches[0]
        used = float(state['accounted_runtime_seconds']) if state.get('closed_cleanly') else \
               float(state['prior_accounted_seconds'])+12*3600
        if not math.isfinite(used) or used<0:
            raise ValueError('Invalid baseline usage ledger')
        prior = max(prior,used)
        for name in (*RESUME_FILES,'per_organ_dice.png'):
            src,dst=path.parent/name,output_root/name
            if src.is_file() and src.resolve()!=dst.resolve():
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,dst)
    return prior


def run_comparison(context, btcv_root, output_root, sam2_checkpoint, sam2_commit, input_root,
                   *, notebook_start=None, work_hours=1.0, minimum_prior_hours=15.1771,
                   total_hours=29.5, resume_root=''):
    if not torch.cuda.is_available():
        raise RuntimeError('Select a Kaggle CUDA GPU for the real baseline')
    if not 0<work_hours<=11.5 or not 0<total_hours<=29.5:
        raise ValueError('Work/session/cumulative budget limits are invalid')
    if not math.isfinite(minimum_prior_hours) or minimum_prior_hours<0:
        raise ValueError('Invalid prior-compute floor')
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    started=time.time() if notebook_start is None else float(notebook_start)
    prior=max(recover_baseline(input_root,root,context,resume_root),minimum_prior_hours*3600)
    accounting_start=started
    ledger=root/'baseline_session_state.json'
    if ledger.is_file():
        old=json.loads(ledger.read_text())
        if old.get('build')==BUILD and old.get('closed_cleanly'):
            finished=float(old.get('finished_unix',started))
            if not math.isfinite(finished) or finished>time.time():
                raise ValueError('Invalid baseline finish timestamp')
            accounting_start=max(started,finished)
    deadline=min(started+work_hours*3600,started+11.5*3600,
                 accounting_start+total_hours*3600-prior-900)
    if deadline<=time.time():
        raise RuntimeError('No baseline time remains within the cumulative/session budget')
    state={'build':BUILD,'checkpoint_sha256':context['reference']['checkpoint_sha256'],
           'closed_cleanly':False,'session_start_unix':started,'accounting_start_unix':accounting_start,
           'prior_accounted_seconds':prior,'work_deadline_unix':deadline,'session_limit_hours':12,
           'total_budget_hours':total_hours}
    write_json(ledger,state)
    status='FAILED';exported=[];result=None
    try:
        trained_cfg=context['reference']['configuration']
        cfg={'sam2':copy.deepcopy(trained_cfg['sam2']),
             'data':copy.deepcopy(trained_cfg['data']),
             'training':{'amp':bool(trained_cfg['training'].get('amp',False))},
             'experiment':{'seed':trained_cfg['experiment']['seed'],'mode':'zero_shot_reference'},
             'baseline':{'training_updates':0,'lora_adapters':False,'posterior_sampling':False,
                         'refinement':False,'prompt_seed':context['seed']}}
        if (sha256_file(sam2_checkpoint)!=cfg['sam2']['checkpoint_sha256']
                or sam2_commit!=cfg['sam2']['resolved_commit']):
            raise ValueError('Frozen SAM2 source/checkpoint differs from the original backbone')
        cfg['sam2']['checkpoint']=str(sam2_checkpoint)
        cfg['data']['root_dir']=str(btcv_root)
        cfg['data']['test_split']=str(root/'test_cases.json')
        cfg['data']['cache_dir']=str(root/'raw_test_cache')
        write_json(root/'baseline_config.json',cfg)
        write_json(root/'frozen_policy.json',context['policy_selection'])
        write_json(root/'test_cases.json',{'cases':context['reference']['splits']['test']})
        write_json(root/'comparison_reference.json',{
            'selected_checkpoint_sha256':context['reference']['checkpoint_sha256'],
            'test_seed':context['seed'],'expected_predictions':context['expected_predictions'],
            'original_balora_no_refinement':context['original_no_refinement'],
            'expected_slice_counts':[{'organ':o,'case_id':c,'n_slices':n} for (o,c),n in context['expected_counts'].items()],
            'audit_source':context['audit_source']})
        set_seed(cfg['experiment']['seed'],deterministic=True)
        d=cfg['data']
        dataset=BTCVDataset(str(btcv_root),str(root/'test_cases.json'),num_classes=d['num_classes'],
                            clip_len=1,train=False,image_size=d['image_size'],ct_window=tuple(d['ct_window']),
                            cache_dir=str(root/'raw_test_cache'),foreground_only=True)
        loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_btcv)
        print('Frozen SAM2 test cases:',dataset.case_ids,'| foreground slices:',len(dataset),flush=True)
        print('Prior accounted hours:',round(prior/3600,4),'| work cap hours:',work_hours,flush=True)
        model=FrozenSAM2(cfg,'cuda:0')
        baseline=run_zero_shot_evaluation(model,loader,'cuda:0',
            test_cases=context['reference']['splits']['test'],seed=context['seed'],
            progress_path=root/'zero_shot_progress.json',deadline=deadline)
        baseline['gpu']=torch.cuda.get_device_name(0);baseline['torch']=torch.__version__
        write_json(root/'zero_shot_evaluation.json',baseline)
        if baseline['complete']:
            result=export_comparison(context,baseline,root)
            status='ZERO_SHOT_COMPARISON_COMPLETE'
            print(json.dumps(result['methods'],indent=2),flush=True)
        else:
            result={'complete':False,'baseline':baseline}
            status='PARTIAL_ZERO_SHOT_BASELINE_RESUME_REQUIRED'
            print('Partial baseline saved; retain the output and resume. No completed comparison claimed.',flush=True)
        exported=[name for name in (*RESUME_FILES,'baseline_config.json','frozen_policy.json',
                  'test_cases.json','comparison_reference.json','per_organ_dice.png')
                  if name!='baseline_session_state.json' and (root/name).is_file()]
        with zipfile.ZipFile(root/'BALoRA_ZeroShot_Comparison_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in exported:z.write(root/name,root.name+'/'+name)
    finally:
        state.update(closed_cleanly=True,accounted_runtime_seconds=prior+max(0,time.time()-accounting_start),
                     finished_unix=time.time(),execution_status=status)
        write_json(ledger,state)
        archive=root/'BALoRA_ZeroShot_Comparison_Results.zip'
        if exported and archive.is_file():
            with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:z.write(ledger,root.name+'/'+ledger.name)
        print('Baseline status:',status,'| cumulative accounted hours:',
              round(state['accounted_runtime_seconds']/3600,4),flush=True)
    return result
