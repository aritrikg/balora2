"""Session recovery regressions; synthetic checkpoints are not research results."""
import json,time,copy
from pathlib import Path
import numpy as np
import pytest,torch
from utils.session_runtime import (session_deadlines,prior_usage,prepare_recovery,
    begin_session,finish_session,training_is_complete,checkpoint_runtime_fields)
from utils.runtime_budget import plan_runtime
from utils.checkpoint import save_checkpoint
from training import train as tr
from tests.test_30h_regressions import tiny_config,patch_tiny
from tests.test_sam2_wrapper_fake import build_fake


def _checkpoint(root,epoch=15,next_batch=280):
    dest=root/'research_fold_01/last.pt';dest.parent.mkdir(parents=True,exist_ok=True)
    cfg=tiny_config(root/'research_fold_01',epochs=20)
    cfg['training'].update(min_epochs=8,early_stopping_patience=6)
    payload={'epoch':epoch,'next_batch':next_batch,'config':cfg,'model_state_dict':{},
        'best_epoch':10,'best_val_dice':0.80174,'patience':5,
        'history':{'epoch':list(range(epoch+1))},'partial_progress':{'updates':next_batch}}
    torch.save(payload,dest)
    return dest,payload


def test_every_session_limits_both_training_and_evaluation_before_platform_timeout():
    p=session_deadlines(1000,0,1000+24.5*3600)
    assert p['training_deadline_unix']==1000+11.5*3600
    assert p['final_deadline_unix']==1000+11.5*3600
    assert p['training_pause_reason']=='SESSION_LIMIT_PAUSED'
    assert p['evaluation_pause_reason']=='SESSION_LIMIT_PAUSED'
    assert p['final_deadline_unix']<1000+12*3600


def test_remaining_total_budget_can_be_stricter_than_a_new_session():
    start=1000;prior=28*3600
    p=session_deadlines(start,prior,start+20*3600)
    assert p['final_deadline_unix']==start+3600
    assert p['training_deadline_unix']<=start+3600
    assert p['evaluation_pause_reason']=='TIME_BUDGET_PAUSED'


def test_legacy_timeout_recovery_charges_the_observed_twelve_hours():
    used,reason=prior_usage({},legacy_hours=12.01)
    assert used==pytest.approx(12.01*3600)
    assert reason=='legacy_timeout_usage_from_supplied_log'


def test_clean_usage_ledger_is_not_charged_twice_on_resume(tmp_path):
    state=begin_session(tmp_path,1000,12.01*3600)
    ended=finish_session(tmp_path,state,'SESSION_PAUSED_RESUME_REQUIRED',now=1000+2*3600)
    used,_=prior_usage({},ended,legacy_hours=12.01)
    assert used==pytest.approx(14.01*3600)
    state2=begin_session(tmp_path,20000,used)
    ended2=finish_session(tmp_path,state2,'FOLD1_PROTOCOL_COMPLETE',now=20000+3600)
    used2,_=prior_usage({},ended2)
    assert used2==pytest.approx(15.01*3600)


def test_unclean_session_is_conservatively_charged_its_platform_limit(tmp_path):
    state=begin_session(tmp_path,1000,12.01*3600)
    used,_=prior_usage({'accounted_runtime_seconds':13*3600},state)
    assert used==pytest.approx(24.01*3600)


def test_checkpoint_only_recovery_does_not_reset_previous_sessions():
    payload=checkpoint_runtime_fields({'session_start_unix':1000,'prior_accounted_seconds':4*3600,
        'session_id':'example','session_limit_hours':12},now=1000+3600)
    assert payload['accounted_runtime_seconds']==5*3600
    used,_=prior_usage(payload)
    assert used==16*3600  # absent a clean ledger, account for the possible full session


def test_default_recovery_requires_attached_weights_instead_of_starting_from_zero(tmp_path):
    with pytest.raises(FileNotFoundError,match='will NOT restart'):
        prepare_recovery(tmp_path/'input',tmp_path/'output','fold1_20epochs_v1')


def test_auto_recovery_copies_only_portable_artifacts_and_preserves_batch(tmp_path):
    source=tmp_path/'input/notebooks/user/balora_v1/42/balora_30h_outputs/fold1_20epochs_v1'
    checkpoint,payload=_checkpoint(source)
    (source/'heldout_progress.json').write_text(json.dumps({'rows':[{'case_id':'img1'}]}))
    output=tmp_path/'working/balora_30h_outputs/fold1_20epochs_v1'
    result=prepare_recovery(tmp_path/'input',output,'fold1_20epochs_v1')
    assert result['completed_epochs']==16 and result['next_batch']==280
    assert result['best_epoch']==10 and result['best_val_dice']==0.80174
    assert not result['training_complete']
    assert result['prior_accounted_seconds']==pytest.approx(12.01*3600)
    assert (output/'research_fold_01/last.pt').read_bytes()==checkpoint.read_bytes()
    assert json.loads((output/'heldout_progress.json').read_text())['rows'][0]['case_id']=='img1'


def test_recovery_rejects_ambiguous_matching_outputs(tmp_path):
    for name in ('run1','run2'):
        _checkpoint(tmp_path/'input'/name/'fold1_20epochs_v1')
    with pytest.raises(ValueError,match='Several matching'):
        prepare_recovery(tmp_path/'input',tmp_path/'out','fold1_20epochs_v1')


def test_saved_early_stop_is_honored_and_partial_epoch_is_not_complete():
    cfg={'training':{'num_epochs':20,'min_epochs':8,'early_stopping_patience':6}}
    p={'config':cfg,'epoch':16,'history':{'epoch':list(range(17))},'patience':6}
    assert training_is_complete(p)
    assert not training_is_complete({**p,'next_batch':50})


def test_completed_early_stop_plans_only_evaluation_without_changing_target():
    p=plan_runtime(2000,500,26*3600,requested_epochs=20,min_epochs=8,
        already_completed=17,training_complete=True,final_seconds_estimate=3600,final_hours=1)
    assert p['requested_protocol_estimated_to_fit']
    assert p['required_additional_epochs']==0 and p['requested_total_epochs']==20


def test_resume_preserves_optimizer_and_session_pause_is_not_completion(tmp_path,monkeypatch):
    patch_tiny(monkeypatch)
    cfg=tiny_config(tmp_path/'run',epochs=1)
    start=time.time()
    cfg['runtime']={'session_start_unix':start,'prior_accounted_seconds':12.01*3600,
        'session_id':'test-session','session_limit_hours':12,
        'training_deadline_unix':start-1,'training_pause_reason':'SESSION_LIMIT_PAUSED'}
    paused=tr.run_training(cfg)
    assert paused['stop_reason']=='SESSION_LIMIT_PAUSED' and not paused['protocol_complete']
    p=tr._read_checkpoint(paused['last_checkpoint'])
    assert p['accounted_runtime_seconds']>=12.01*3600
    cfg['runtime']['training_deadline_unix']=time.time()+60
    completed=tr.run_training(cfg,resume_from=paused['last_checkpoint'])
    assert completed['protocol_complete']
    saved=tr._read_checkpoint(completed['last_checkpoint'])
    assert saved['training_protocol_complete']
    assert all(int(s['step'])==1 for s in saved['optimizer_state_dict']['state'].values() if 'step' in s)


def test_resume_of_early_stopped_run_performs_no_new_optimizer_update(tmp_path,monkeypatch):
    patch_tiny(monkeypatch)
    cfg=tiny_config(tmp_path/'first',epochs=1)
    result=tr.run_training(cfg)
    saved=tr._read_checkpoint(result['last_checkpoint'])
    # Represent a completed early stop under a longer, unchanged protocol.
    cfg['training'].update(num_epochs=20,min_epochs=1,early_stopping_patience=6)
    saved['config']=copy.deepcopy(cfg);saved['patience']=6
    saved['training_protocol_complete']=True
    cp=tmp_path/'early.pt';torch.save(saved,cp)
    def fail(*args,**kwargs):raise AssertionError('Completed early stopping was restarted')
    monkeypatch.setattr(tr,'train_one_epoch',fail)
    cfg['experiment']['output_dir']=str(tmp_path/'continued')
    out=tr.run_training(cfg,resume_from=str(cp))
    assert out['protocol_complete'] and out['completed_epochs']==1
    assert out['stop_reason']=='EARLY_STOPPING_COMPLETED'


def test_validation_session_pause_does_not_repeat_completed_training_updates(tmp_path,monkeypatch):
    patch_tiny(monkeypatch)
    import evaluation.evaluate as ev
    actual_evaluation=ev.run_evaluation
    monkeypatch.setattr(ev,'run_evaluation',lambda *a,**kw:{'protocol':{'complete':False}})
    cfg=tiny_config(tmp_path/'paused_validation',epochs=1)
    cfg['runtime']={'training_pause_reason':'SESSION_LIMIT_PAUSED'}
    paused=tr.run_training(cfg)
    saved=tr._read_checkpoint(paused['last_checkpoint'])
    assert paused['stop_reason']=='SESSION_LIMIT_PAUSED'
    assert saved['next_batch']==1 and saved['partial_progress']['updates']==1
    monkeypatch.setattr(ev,'run_evaluation',actual_evaluation)
    def no_replay(*args,**kwargs):raise AssertionError('Completed training batch was replayed')
    monkeypatch.setattr(tr,'train_one_epoch',no_replay)
    out=tr.run_training(cfg,resume_from=paused['last_checkpoint'])
    assert out['protocol_complete']
    final=tr._read_checkpoint(out['last_checkpoint'])
    assert all(int(s['step'])==1 for s in final['optimizer_state_dict']['state'].values() if 'step' in s)
