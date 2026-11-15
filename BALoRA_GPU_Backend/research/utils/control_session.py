"""Recover a control bundle and preserve total and per-session wall time."""
from __future__ import annotations
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import time
import uuid
import zipfile
from evaluation.control_inputs import BUILD
from utils.runtime_budget import write_json

LEDGER='control_session_state.json'
ARCHIVE='BALoRA_Deterministic_LoRA_Control_Results.zip'
TOP_FILES={LEDGER,'last.pt','best.pt','control_config.json','frozen_protocol.json',
    'training_history.json','training_status.json','profile_report.json','runtime_plan.json',
    'selected_control_checkpoint.json','control_test_progress.json','control_test_evaluation.json',
    'method_comparison.json','method_comparison.csv','per_organ_comparison.csv',
    'per_volume_comparison.csv','per_organ_dice.png','failure.json','control_status.json',
    'architecture_report.json','runtime_execution.json','figure_manifest.json'}


def allowed_file(relative):
    p=PurePosixPath(relative)
    return len(p.parts)==1 and relative in TOP_FILES or (
        len(p.parts)==2 and ((p.parts[0] in ('validation_reports','splits') and p.suffix=='.json') or (p.parts[0]=='figures' and p.suffix in ('.png','.svg','.pdf','.csv'))))


def zip_members(source):
    with zipfile.ZipFile(source) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
            raise ValueError('Unsafe or duplicate control ZIP entries')
        hits=[n for n in names if n.endswith(LEDGER)]
        if len(hits)!=1:raise ValueError('Expected one control session ledger')
        prefix=hits[0][:-len(LEDGER)]
        return prefix,json.loads(z.read(hits[0]))


def recover_control(input_root,output_root,protocol_sha256,explicit=''):
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    if (root/LEDGER).is_file():
        source=root
    elif explicit:
        source=Path(explicit).expanduser().resolve()
    else:
        found=[p.parent for p in Path(input_root).rglob(LEDGER)]
        if not found:
            found=[p for p in Path(input_root).rglob('*.zip')
                   if p.name.lower().startswith('balora_deterministic_lora_control_results')]
        if len(found)>1:
            # Multiple profiler-only retries have no research weights to
            # choose between. Retain the greatest clean accounted usage.
            rejected=[]
            for p in found:
                if p.is_file():
                    prefix,saved=zip_members(p)
                    with zipfile.ZipFile(p) as z:
                        weights=any(prefix+n in z.namelist() for n in ('last.pt','best.pt'))
                else:
                    saved=json.loads((p/LEDGER).read_text())
                    weights=any((p/n).is_file() for n in ('last.pt','best.pt'))
                if (not weights and saved.get('closed_cleanly')
                        and saved.get('execution_status')=='BUDGET_PLAN_REJECTED'
                        and saved.get('build')==BUILD and saved.get('protocol_sha256')==protocol_sha256):
                    rejected.append((float(saved['accounted_runtime_seconds']),p))
            if len(rejected)!=len(found):
                raise ValueError('Several control outputs found; set CONTROL_RESUME_ROOT')
            found=[max(rejected,key=lambda x:x[0])[1]]
        if not found:return None
        source=found[0]
    if source.is_file():
        prefix,state=zip_members(source)
    else:
        state=json.loads((source/LEDGER).read_text())
    if state.get('build')!=BUILD or state.get('protocol_sha256')!=protocol_sha256:
        raise ValueError('Control recovery belongs to a different protocol')
    if source.resolve()!=root.resolve():
        if (root/'last.pt').exists():raise FileExistsError('Refusing to overwrite a local control checkpoint')
        if source.is_file():
            with zipfile.ZipFile(source) as z:
                for n in z.namelist():
                    rel=n[len(prefix):] if n.startswith(prefix) else ''
                    if allowed_file(rel):
                        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                        with z.open(n) as src,dest.open('wb') as out:shutil.copyfileobj(src,out)
        else:
            for src in source.rglob('*'):
                if src.is_file() and allowed_file(src.relative_to(source).as_posix()):
                    dest=root/src.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(src,dest)
    return state


def begin_control_session(input_root,output_root,protocol_sha256,reference_prior,*,
        notebook_start,process_token,work_hours=11.,minimum_prior_hours=15.6244,
        total_hours=29.5,resume_root='',now=None):
    now=time.time() if now is None else float(now)
    vals=(notebook_start,work_hours,minimum_prior_hours,total_hours,reference_prior)
    if any(not math.isfinite(float(x)) for x in vals) or not 0<work_hours<=11.5 or not 0<total_hours<=29.5 or reference_prior<0 or minimum_prior_hours<0 or notebook_start>now:
        raise ValueError('Invalid control session budget')
    old=recover_control(input_root,output_root,protocol_sha256,resume_root)
    prior=max(float(reference_prior),minimum_prior_hours*3600)
    session_start=float(notebook_start);accounting_start=session_start
    reason='new_control'
    if old:
        same=old.get('process_token')==process_token
        if same and not old.get('closed_cleanly'):
            prior=max(prior,float(old['prior_accounted_seconds']))
            session_start=float(old['session_start_unix'])
            accounting_start=float(old['accounting_start_unix'])
            reason='same_kernel_active_session'
        else:
            used=float(old['accounted_runtime_seconds']) if old.get('closed_cleanly') else float(old['prior_accounted_seconds'])+12*3600
            prior=max(prior,used)
            if same:
                session_start=float(old['session_start_unix'])
                accounting_start=max(float(notebook_start),float(old['finished_unix']))
                reason='same_kernel_clean_retry'
            else:reason='clean_resume' if old.get('closed_cleanly') else 'unclean_resume_charged_12h'
    if not math.isfinite(prior) or prior<0:raise ValueError('Invalid saved control usage')
    session_work=session_start+work_hours*3600
    cumulative_work=accounting_start+total_hours*3600-prior-900
    deadline=min(session_work,cumulative_work)
    if deadline<=now:raise ValueError('No control work remains within the cumulative/session budget')
    state={'build':BUILD,'protocol_sha256':protocol_sha256,'process_token':process_token,
        'closed_cleanly':False,'session_id':uuid.uuid4().hex,'session_start_unix':session_start,
        'accounting_start_unix':accounting_start,'prior_accounted_seconds':prior,
        'accounted_runtime_seconds':prior+max(0,now-accounting_start),
        'work_deadline_unix':deadline,'cumulative_work_deadline_unix':cumulative_work,
        'session_limit_hours':12,'work_cap_hours':work_hours,'total_budget_hours':total_hours,
        'export_reserve_seconds':900,'accounting_reason':reason,'execution_status':'ACTIVE'}
    write_json(Path(output_root)/LEDGER,state)
    return state


def finish_control_session(output_root,state,status,now=None):
    now=time.time() if now is None else float(now)
    state=dict(state)
    state.update(closed_cleanly=True,finished_unix=now,execution_status=status,
        accounted_runtime_seconds=state['prior_accounted_seconds']+max(0.,now-state['accounting_start_unix']))
    write_json(Path(output_root)/LEDGER,state)
    return state


def export_control_bundle(output_root):
    root=Path(output_root);target=root/ARCHIVE
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and allowed_file(p.relative_to(root).as_posix()) and p.name!=LEDGER:
                z.write(p,root.name+'/'+p.relative_to(root).as_posix())
    return target


def append_clean_ledger(archive,output_root):
    root=Path(output_root)
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
        z.write(root/LEDGER,root.name+'/'+LEDGER)
