import copy,json,time,zipfile
from pathlib import Path
import pytest,torch
from tests.test_deterministic_control import cfg,model,loader
from training.deterministic_control import deterministic_objective,control_weights,profile_control
from evaluation.control_inputs import protocol_digest
from utils.full_accounting import reconciled_prior_hours


def test_activation_policy_preserves_scientific_protocol_and_gradients(tmp_path):
    c=cfg(tmp_path);c['lora']['dropout']=0.
    a=model(c)
    d=copy.deepcopy(c);d['runtime']={'encoder_checkpointing':'off'}
    b=model(d);b.load_state_dict(a.state_dict())
    assert protocol_digest(c)==protocol_digest(d)
    assert not b.encoder_gradient_checkpointing_enabled
    batch=next(iter(loader(['t1'])))
    outputs=[];gradients=[]
    for m in (a,b):
        m.train();torch.manual_seed(88)
        loss,_=deterministic_objective(m,batch['image'][:,0],batch['label'][:,0],control_weights(c),1,False,1)
        loss.backward();outputs.append(loss.detach());gradients.append([p.grad.clone() for p in m.parameters() if p.requires_grad])
    torch.testing.assert_close(*outputs)
    for x,y in zip(*gradients):torch.testing.assert_close(x,y)


def test_profiling_oom_falls_back_without_reusing_weights(tmp_path,monkeypatch):
    import training.control_execution as module
    c=cfg(tmp_path);c['runtime']={'encoder_checkpointing':'off'};seen=[]
    def profile(*a,**kw):
        seen.append(c['runtime']['encoder_checkpointing'])
        if len(seen)==1:raise torch.cuda.OutOfMemoryError('test fixture')
        return {'complete':True}
    monkeypatch.setattr(module,'profile_control',profile)
    assert module.profile_execution(c,None,None,'cpu',deadline=time.time()+10)['complete']
    assert seen==['off','per_block']


def test_checkpointed_profile_oom_is_not_swallowed(tmp_path,monkeypatch):
    import training.control_execution as module
    c=cfg(tmp_path)
    def fail(*a,**k):raise torch.cuda.OutOfMemoryError('test fixture')
    monkeypatch.setattr(module,'profile_control',fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):module.profile_execution(c,None,None,'cpu',deadline=time.time()+10)


def test_activation_mode_invalidates_profile_cache(tmp_path,monkeypatch):
    import training.deterministic_control as module
    c=cfg(tmp_path);c['runtime']={'encoder_checkpointing':'off'}
    (tmp_path/'profile_report.json').write_text(json.dumps({'complete':True,'protocol_sha256':c['control']['protocol_sha256'],'gpu':'CPU_TEST','runtime_revision':module.RUNTIME_REVISION,'execution_mode':'per_block'}))
    def fresh(*a,**kw):raise RuntimeError('fresh hardware measurement')
    monkeypatch.setattr(module,'DeterministicSAM2',fresh)
    with pytest.raises(RuntimeError,match='fresh hardware'):profile_control(c,None,None,'cpu',deadline=time.time()+10)


def test_accounting_reconciles_omitted_nonoverlapping_attempt_and_deduplicates(tmp_path):
    for i,(start,end,prior) in enumerate([(100,110,1000),(200,220,1010),(300,330,1010)]):
        s={'protocol_sha256':'p','closed_cleanly':True,'session_id':str(i),'accounting_start_unix':start,'finished_unix':end,'prior_accounted_seconds':prior}
        with zipfile.ZipFile(tmp_path/f'BALoRA_Deterministic_LoRA_Control_Results({i}).zip','w') as z:z.writestr('output/control_session_state.json',json.dumps(s))
    folder=tmp_path/'copy';folder.mkdir();(folder/'control_session_state.json').write_text(json.dumps(s))
    assert reconciled_prior_hours(tmp_path,'p',0)==pytest.approx(1060/3600)


def test_figures_are_included_in_portable_export(tmp_path):
    from utils.control_session import export_control_bundle
    (tmp_path/'figures').mkdir();(tmp_path/'figures'/'figure_1.png').write_bytes(b'test fixture')
    (tmp_path/'figure_manifest.json').write_text('{}')
    with zipfile.ZipFile(export_control_bundle(tmp_path)) as z:
        assert any(n.endswith('/figures/figure_1.png') for n in z.namelist())
        assert any(n.endswith('/figure_manifest.json') for n in z.namelist())
