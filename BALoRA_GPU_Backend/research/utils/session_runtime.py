"""Portable Kaggle-session recovery and separate total/session time limits.

No cloud API or credentials are used. Only attached artifacts from this run
are copied; checkpoints remain user-owned and model/split identity is checked
by the training code before any update.
"""
from __future__ import annotations
import json, math, os, shutil, tempfile, time, uuid
from pathlib import Path
from utils.runtime_budget import write_json

STATE_NAME = 'session_state.json'
COPY_FILES = (
    'research_fold_01/last.pt', 'research_fold_01/best.pt',
    'research_fold_01/training_history.json', 'research_fold_01/training_status.json',
    'runtime_budget_report.json', 'heldout_progress.json', 'heldout_evaluation.json',
    'selected_checkpoint.json', 'experiment_status.json', STATE_NAME,
)


def read_payload(path):
    import torch
    try:
        return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError:
        return torch.load(path,map_location='cpu')


def training_is_complete(payload):
    """Honor a saved early stop or target completion, including v2 files."""
    if int(payload.get('next_batch',0)):
        return False
    if payload.get('training_protocol_complete'):
        return True
    cfg=payload.get('config',{});tr=cfg.get('training',{})
    epochs=payload.get('history',{}).get('epoch',[])
    completed=len(epochs) if epochs else max(0,int(payload.get('epoch',-1))+1)
    target=int(tr.get('num_epochs',20));minimum=int(tr.get('min_epochs',8))
    return completed>=target or (completed>=minimum and
        int(payload.get('patience',0))>=int(tr.get('early_stopping_patience',6)))


def accounted_seconds(runtime, now=None):
    now=time.time() if now is None else float(now)
    start=float(runtime.get('session_start_unix',now))
    return float(runtime.get('prior_accounted_seconds',0))+max(0.0,now-start)


def checkpoint_runtime_fields(runtime, now=None):
    if 'session_start_unix' not in runtime:
        return {}
    return {'accounted_runtime_seconds':accounted_seconds(runtime,now),
        'runtime_session_id':runtime.get('session_id'),
        'runtime_session_prior_seconds':float(runtime.get('prior_accounted_seconds',0)),
        'runtime_session_limit_hours':float(runtime.get('session_limit_hours',12)),
        'runtime_session_closed_cleanly':False}


def session_deadlines(session_start, prior_seconds, global_training_deadline,
                      total_budget_hours=29.5,total_safety_hours=0.5,
                      session_limit_hours=12.0,save_margin_hours=0.5):
    vals=(session_start,prior_seconds,global_training_deadline,total_budget_hours,
          total_safety_hours,session_limit_hours,save_margin_hours)
    if any(not math.isfinite(float(v)) for v in vals):
        raise ValueError('Session timing values must be finite')
    if prior_seconds<0 or not 0<save_margin_hours<session_limit_hours:
        raise ValueError('Invalid prior time or session save margin')
    if not 0<=total_safety_hours<total_budget_hours:
        raise ValueError('Invalid total safety reserve')
    session_work=session_start+(session_limit_hours-save_margin_hours)*3600
    total_final=session_start-prior_seconds+(total_budget_hours-total_safety_hours)*3600
    training=min(global_training_deadline,session_work,total_final)
    final=min(session_work,total_final)
    return {'training_deadline_unix':training,'final_deadline_unix':final,
        'global_training_deadline_unix':global_training_deadline,'global_final_deadline_unix':total_final,
        'session_work_deadline_unix':session_work,
        'training_pause_reason':'SESSION_LIMIT_PAUSED' if session_work<=min(global_training_deadline,total_final) else 'TIME_BUDGET_PAUSED',
        'evaluation_pause_reason':'SESSION_LIMIT_PAUSED' if session_work<=total_final else 'TIME_BUDGET_PAUSED',
        'session_limit_hours':session_limit_hours,'session_save_margin_hours':save_margin_hours}


def prior_usage(payload,state=None,legacy_hours=12.01,minimum_hours=0):
    """Do not reset cumulative runtime when the host/kernel changes.

    Clean v3 output records exact consumed time. An unclosed v3 session is
    conservatively charged its full declared session limit. V2 has no usage
    ledger; the recovery preset charges the 12.01 h observed in the user's log.
    """
    if not math.isfinite(legacy_hours) or legacy_hours<0 or minimum_hours<0:
        raise ValueError('Prior usage must be finite and non-negative')
    state=state or {}
    if state.get('closed_cleanly'):
        used=float(state['accounted_runtime_seconds'])
        reason='closed_session_ledger'
    elif state.get('session_id'):
        used=float(state['prior_accounted_seconds'])+float(state.get('session_limit_hours',12))*3600
        reason='unclosed_session_charged_full_limit'
    elif payload.get('runtime_session_id'):
        used=float(payload.get('runtime_session_prior_seconds',0))+float(payload.get('runtime_session_limit_hours',12))*3600
        reason='checkpoint_only_unclosed_session_charged_full_limit'
    else:
        used=legacy_hours*3600
        reason='legacy_timeout_usage_from_supplied_log'
    used=max(used,float(payload.get('accounted_runtime_seconds',0)),minimum_hours*3600)
    return used,reason


def _atomic_copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.restore');os.close(fd)
    try:
        shutil.copyfile(source,tmp);os.replace(tmp,dest)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def prepare_recovery(input_root,output_root,run_id,action='resume',resume_from='',
                     resume_root='',legacy_hours=12.01,minimum_prior_hours=0):
    output=Path(output_root).resolve();input_root=Path(input_root)
    output.mkdir(parents=True,exist_ok=True)
    if action not in ('resume','fresh'):
        raise ValueError('BALORA_ACTION must be resume or fresh')
    local=output/'research_fold_01/last.pt'
    if action=='fresh':
        if local.exists() or (output/'research_fold_01/best.pt').exists():
            raise FileExistsError('Fresh run output already contains trained weights; resume it or choose a new BALORA_RUN_ID')
        return {'resume_from':'','source_root':None,'prior_accounted_seconds':minimum_prior_hours*3600,
                'training_complete':False,'completed_epochs':0,'next_batch':0,'accounting_reason':'fresh_run'}
    if resume_from:
        source=Path(resume_from).expanduser().resolve()
        if not source.is_file():raise FileNotFoundError(f'Resume checkpoint not found: {source}')
    elif resume_root:
        source=Path(resume_root).expanduser().resolve()/'research_fold_01/last.pt'
        if not source.is_file():raise FileNotFoundError(f'No research_fold_01/last.pt under {resume_root}')
    elif local.is_file():
        source=local
    else:
        candidates=[]
        if input_root.is_dir():
            for path in input_root.rglob('last.pt'):
                if path.parent.name=='research_fold_01' and path.parent.parent.name==run_id:
                    candidates.append(path.resolve())
        candidates=sorted(set(candidates))
        if not candidates:
            raise FileNotFoundError(
                'Recovery mode requires the saved Version 11 output. In Kaggle Add Input, attach that notebook output '
                'containing '+run_id+'/research_fold_01/last.pt, or set BALORA_RESUME_FROM to the actual last.pt path. '
                'This run will NOT restart training from scratch. For an intentional new experiment set BALORA_ACTION=fresh.')
        if len(candidates)!=1:
            raise ValueError('Several matching last.pt files found; set BALORA_RESUME_FROM explicitly: '+str(candidates))
        source=candidates[0]
    source_root=source.parent.parent if source.parent.name=='research_fold_01' else source.parent
    if source!=local.resolve() and local.exists():
        raise FileExistsError('Local output already has a different last.pt; do not overwrite it. Choose a clean output directory.')
    state_path=source_root/STATE_NAME
    state=json.loads(state_path.read_text()) if state_path.is_file() else {}
    payload=read_payload(source)
    if not isinstance(payload.get('config'),dict) or 'model_state_dict' not in payload:
        raise ValueError('This is not a resumable BALoRA training checkpoint')
    used,reason=prior_usage(payload,state,legacy_hours,minimum_prior_hours)
    completed=len(payload.get('history',{}).get('epoch',[]))
    result={'resume_from':str(local),'source_root':str(source_root),
        'prior_accounted_seconds':used,'accounting_reason':reason,
        'training_complete':training_is_complete(payload),'completed_epochs':completed,
        'next_batch':int(payload.get('next_batch',0)),'saved_config':payload['config'],'best_epoch':payload.get('best_epoch'),
        'best_val_dice':payload.get('best_val_dice')}
    if source_root!=output:
        for rel in COPY_FILES:
            original=source_root/rel
            if original.is_file():_atomic_copy(original,output/rel)
        if not local.is_file():_atomic_copy(source,local)
    # Reuse previous timing measurements; split/model compatibility is checked
    # after the real dataset's split has been resolved and before allocation.
    timing=output/'runtime_budget_report.json'
    result['runtime_benchmark']=json.loads(timing.read_text()) if timing.is_file() else None
    # Read-only NIfTI cache can be reused without copying multiple gigabytes.
    for parent in (source_root,)+tuple(source_root.parents):
        candidate=parent/'balora_raw_cache'
        if candidate.is_dir():
            result['read_only_cache_root']=str(candidate);break
        if parent==input_root:break
    del payload
    return result


def begin_session(output_root,session_start,prior_seconds,session_limit_hours=12.0,*,persist=True):
    state={'schema_version':3,'session_id':uuid.uuid4().hex,'closed_cleanly':False,
        'session_start_unix':session_start,'prior_accounted_seconds':prior_seconds,
        'accounted_runtime_seconds':prior_seconds,'session_limit_hours':session_limit_hours}
    if persist:write_json(Path(output_root)/STATE_NAME,state)
    return state


def finish_session(output_root,session_state,execution_status,now=None):
    state=dict(session_state);now=time.time() if now is None else now
    state.update(closed_cleanly=True,finished_unix=now,execution_status=execution_status,
        accounted_runtime_seconds=state['prior_accounted_seconds']+max(0,now-state['session_start_unix']))
    write_json(Path(output_root)/STATE_NAME,state)
    return state
