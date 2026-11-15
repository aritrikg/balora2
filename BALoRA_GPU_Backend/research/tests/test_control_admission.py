import copy,json,time,zipfile
import pytest
from tests.test_deterministic_control import cfg,loader
from training.deterministic_control import plan_control,profile_control,RUNTIME_REVISION
from utils.control_profile import stage_saved_profile,validate_profile
from utils.control_session import recover_control,LEDGER
from evaluation.control_inputs import BUILD


def record(c):
    return {'profile':{'complete':True,'runtime_revision':RUNTIME_REVISION,
        'protocol_sha256':c['control']['protocol_sha256'],'gpu':'CPU_TEST',
        'training_sample_seconds':[.1,.2],'training_sample_cases':['t1','t2'],
        'training_p75_seconds_per_slice':.2,'train_foreground_slices':4,
        'validation_foreground_slices':4,'validation_sample_slices':2,'validation_sample_seconds':.2},
        'source_archive_sha256':'a'*64,'accounted_runtime_seconds':120.,
        'source_execution_status':'BUDGET_PLAN_REJECTED'}


def test_latest_measured_t4_profile_passes_with_explicit_twenty_percent_margin():
    p={'training_p75_seconds_per_slice':1.241555614000049,'train_foreground_slices':1347,
        'validation_sample_seconds':18.87906289100647,'validation_sample_slices':30,
        'validation_foreground_slices':371}
    now=100000.
    state={'work_deadline_unix':now+11*3600,
           'cumulative_work_deadline_unix':now+47252.2796895504+1800}
    result=plan_control(p,state,996.0540652275085,now=now)
    assert result['requested_protocol_estimated_to_fit']
    assert result['epoch_margin']==.2 and result['requested_total_epochs']==20
    assert result['estimated_training_seconds']/3600==pytest.approx(12.705643265401196)
    assert result['estimated_final_evaluation_reserve_seconds']==1800
    assert result['training_deadline_unix']<=state['work_deadline_unix']
    assert result['final_deadline_unix']==state['work_deadline_unix']


def test_saved_profile_bypasses_discarded_model_and_optimizer_updates(tmp_path,monkeypatch):
    from training import deterministic_control as module
    c=cfg(tmp_path);r=record(c)
    staged=stage_saved_profile(tmp_path,c,r,120.)
    assert staged['staged']
    def forbidden(*a,**k):raise AssertionError('Discarded diagnostic model was reconstructed')
    monkeypatch.setattr(module,'DeterministicSAM2',forbidden)
    profile=profile_control(c,loader(['t1','t2']),loader(['v1','v2']),'cpu',deadline=time.time()+30)
    assert profile['training_sample_seconds']==r['profile']['training_sample_seconds']
    assert profile['cache_provenance']['source_archive_sha256']=='a'*64
    assert not (tmp_path/'last.pt').exists()


def test_cached_measurement_from_another_gpu_is_not_used(tmp_path,monkeypatch):
    from training import deterministic_control as module
    c=cfg(tmp_path);r=record(c);r['profile']['gpu']='Tesla T4'
    stage_saved_profile(tmp_path,c,r,120.)
    def different_hardware(*a,**k):raise RuntimeError('fresh profile required')
    monkeypatch.setattr(module,'DeterministicSAM2',different_hardware)
    with pytest.raises(RuntimeError,match='fresh profile'):
        profile_control(c,loader(['t1','t2']),loader(['v1','v2']),'cpu',deadline=time.time()+30)


def test_profile_reuse_cannot_reset_prior_accounting(tmp_path):
    c=cfg(tmp_path)
    with pytest.raises(ValueError,match='accounting'):
        stage_saved_profile(tmp_path,c,record(c),119.)


@pytest.mark.parametrize('change', ['protocol','samples','p75','nonfinite'])
def test_profile_cache_rejects_invalid_provenance_or_measurements(tmp_path,change):
    c=cfg(tmp_path);r=record(c);p=r['profile']
    if change=='protocol':p['protocol_sha256']='another'
    elif change=='samples':p['training_sample_cases']=['test_case','t2']
    elif change=='p75':p['training_p75_seconds_per_slice']=.15
    else:p['validation_sample_seconds']=float('nan')
    with pytest.raises(ValueError):stage_saved_profile(tmp_path,c,r,120.)


def test_existing_compatible_profile_is_preserved(tmp_path):
    c=cfg(tmp_path);r=record(c)
    stage_saved_profile(tmp_path,c,r,120.)
    current=json.loads((tmp_path/'profile_report.json').read_text())
    current['gpu']='a different cached device'
    (tmp_path/'profile_report.json').write_text(json.dumps(current))
    assert not stage_saved_profile(tmp_path,c,r,120.)['staged']
    assert json.loads((tmp_path/'profile_report.json').read_text())['gpu']==current['gpu']


def test_raw_results_zip_with_download_suffix_is_discovered(tmp_path):
    inputs=tmp_path/'input';inputs.mkdir()
    path=inputs/'BALoRA_Deterministic_LoRA_Control_Results(1).zip'
    state={'build':BUILD,'protocol_sha256':'identity','closed_cleanly':True,
           'accounted_runtime_seconds':500.,'execution_status':'BUDGET_PLAN_REJECTED'}
    with zipfile.ZipFile(path,'w') as z:z.writestr('output/'+LEDGER,json.dumps(state))
    result=recover_control(inputs,tmp_path/'restored','identity')
    assert result['accounted_runtime_seconds']==500.


def test_multiple_rejected_profiles_retain_greatest_accounted_usage(tmp_path):
    inputs=tmp_path/'input';inputs.mkdir()
    for name,used in [('first',500.),('latest',700.)]:
        p=inputs/name;p.mkdir()
        (p/LEDGER).write_text(json.dumps({'build':BUILD,'protocol_sha256':'identity','closed_cleanly':True,
            'accounted_runtime_seconds':used,'execution_status':'BUDGET_PLAN_REJECTED'}))
    assert recover_control(inputs,tmp_path/'restored','identity')['accounted_runtime_seconds']==700.
    (inputs/'latest/last.pt').write_bytes(b'branch checkpoint')
    with pytest.raises(ValueError,match='Several control outputs'):
        recover_control(inputs,tmp_path/'another','identity')
