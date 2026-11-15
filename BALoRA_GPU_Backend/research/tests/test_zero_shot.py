import copy
import json
from pathlib import Path
import time
import zipfile

import pytest
import torch
from torch.utils.data import Dataset,DataLoader
from models.frozen_sam2 import FrozenSAM2
from tests.fake_sam2 import FakeSAM2
from tests.test_sam2_wrapper_fake import build_fake
from tests.test_training_and_refinement_fake import OneClip,collate
from tests.test_validation_refinement_audit import saved_config
from evaluation.zero_shot import (BUILD, AUDIT_FILES, export_comparison, locate_audit,
                                  locate_original_results, read_bundle, run_zero_shot_evaluation)
from evaluation.evaluate import _prediction_seed
from utils.prompting import sample_foreground_points


def cfg():
    c=saved_config()
    c['sam2']={'model_cfg':'fake','checkpoint':'fake','checkpoint_sha256':'test-only','resolved_commit':'fake'}
    c['training']={'amp':False}
    c['data']={'num_classes':2,'image_size':64,'ct_window':[-175,250]}
    return c


class TestClips(Dataset):
    __test__=False
    def __init__(self,c):
        self.case_ids=c['experiment']['split_case_ids']['test']
        self._clip_index=[(cid,16,17) for cid in self.case_ids]
    def __len__(self):return len(self.case_ids)
    def __getitem__(self,i):
        row=OneClip()[0];row['case_id']=self.case_ids[i]
        return row


def setup():
    c=cfg();dataset=TestClips(c)
    return c,FrozenSAM2(c,'cpu',backbone=FakeSAM2(image_size=64)),DataLoader(dataset,batch_size=1,collate_fn=collate)


def evaluate(model,loader,c,path,deadline=None):
    return run_zero_shot_evaluation(model,loader,'cpu',
        test_cases=c['experiment']['split_case_ids']['test'],seed=700043,
        progress_path=path,deadline=deadline)


def test_native_backbone_is_frozen_without_adapters_or_rank_controller():
    _,model,_=setup()
    assert not any(p.requires_grad for p in model.parameters())
    assert not hasattr(model,'rank_controller')
    assert not any('lora' in name for name,_ in model.named_parameters())


def test_rejects_a_backbone_with_injected_trained_adapters():
    with pytest.raises(ValueError,match='no injected LoRA'):
        FrozenSAM2(cfg(),'cpu',backbone=build_fake().backbone)


def test_native_predictions_equal_existing_decoder_with_zero_adapter_means():
    c=cfg()
    adapted=build_fake();adapted.eval()
    # Recover the same native parameter values from the fresh fake wrapper.
    fresh=copy.deepcopy(adapted.backbone)
    from models.bayesian_lora import BayesianLoRALinear, DeterministicLoRALinear
    for parent in list(fresh.modules()):
        for name,child in list(parent.named_children()):
            if isinstance(child,(BayesianLoRALinear,DeterministicLoRALinear)):
                setattr(parent,name,child.base_layer)
    native=FrozenSAM2(c,'cpu',backbone=fresh)
    image=torch.rand(1,1,64,64);points=torch.tensor([[[23.,30.]]]);labels=torch.ones(1,1,dtype=torch.int32)
    with torch.no_grad():
        native_logits=native.decode_one(native.encode_image(image),points,labels)
        adapted_logits=adapted.decode(adapted.encode_image(image),points,labels,k_samples=1,sample=False)[0]
    assert torch.allclose(native_logits,adapted_logits,atol=1e-6,rtol=0)


def test_test_prediction_coverage_prompt_rng_and_frozen_weights(tmp_path):
    c,model,loader=setup()
    state={n:p.clone() for n,p in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    report=evaluate(model,loader,c,tmp_path/'progress.json')
    assert report['complete'] and report['protocol']['n_scored_slice_organ_predictions']==6
    assert report['protocol']['posterior_samples'] is None
    assert report['protocol']['epistemic_uncertainty_estimated'] is False
    assert torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(v,state[n]) for n,v in model.state_dict().items())
    progress=json.loads((tmp_path/'progress.json').read_text())
    target=OneClip()[0]['label'][0,1].unsqueeze(0)
    for row in progress['rows']:
        torch.manual_seed(_prediction_seed(700043,row['case_id'],row['slice_index'],row['organ_class']))
        expected=sample_foreground_points(target)[0,0].tolist()
        assert row['prompt_xy']==expected


def test_completed_progress_never_decodes_the_same_prediction_again(tmp_path,monkeypatch):
    c,model,loader=setup();path=tmp_path/'progress.json'
    first=evaluate(model,loader,c,path)
    def forbidden(*args,**kwargs):raise AssertionError('Repeated completed prediction')
    monkeypatch.setattr(model,'decode_one',forbidden)
    second=evaluate(model,loader,c,path)
    assert first['segmentation']==second['segmentation']


def test_deadline_pause_is_partial_and_can_resume(tmp_path):
    c,model,loader=setup();path=tmp_path/'progress.json'
    partial=evaluate(model,loader,c,path,deadline=time.time()-1)
    assert not partial['complete']
    assert not json.loads(path.read_text())['complete']
    assert evaluate(model,loader,c,path)['complete']


def test_changed_seed_cannot_reuse_other_progress(tmp_path):
    c,model,loader=setup();path=tmp_path/'progress.json';evaluate(model,loader,c,path)
    with pytest.raises(ValueError,match='another backbone/data/protocol'):
        run_zero_shot_evaluation(model,loader,'cpu',test_cases=loader.dataset.case_ids,
                                 seed=999,progress_path=path)


def test_non_test_cases_are_blocked_before_decoding(tmp_path):
    c,model,loader=setup()
    with pytest.raises(ValueError,match='Only the saved test'):
        run_zero_shot_evaluation(model,loader,'cpu',test_cases=['validation_case'],
                                 seed=700043,progress_path=tmp_path/'p.json')


def comparison_context(c,report):
    return {'expected_counts':{(organ,cid):r['n_slices'] for organ,cases in report['per_volume_segmentation'].items() for cid,r in cases.items()},
            'original_no_refinement':copy.deepcopy(report),'reference':{'checkpoint_sha256':'adapter-test','splits':{'test':c['experiment']['split_case_ids']['test']}},
            'seed':700043,'expected_predictions':6}


def test_comparison_checks_matched_coverage_and_writes_tables(tmp_path):
    c,model,loader=setup();report=evaluate(model,loader,c,tmp_path/'p.json')
    context=comparison_context(c,report)
    result=export_comparison(context,report,tmp_path/'tables')
    assert result['complete'] and result['identical_case_organ_slice_coverage_verified']
    assert result['methods'][0]['dice']==result['methods'][1]['dice']
    assert (tmp_path/'tables/per_volume_comparison.csv').is_file()
    context['expected_counts'][('spleen','missing')]=1
    with pytest.raises(ValueError,match='identical volume/organ'):
        export_comparison(context,report,tmp_path/'bad')


def test_partial_results_cannot_claim_a_completed_comparison(tmp_path):
    with pytest.raises(ValueError,match='partial baseline'):
        export_comparison({}, {'complete':False},tmp_path)


def test_prefixed_audit_archive_is_discovered_and_read(tmp_path):
    p=tmp_path/'BALoRA_validation_audit_results.zip'
    with zipfile.ZipFile(p,'w') as z:
        for name in AUDIT_FILES:z.writestr('audit_folder/'+name,json.dumps({'fixture':True}))
    assert locate_audit(tmp_path)==p
    assert all(v['fixture'] for v in read_bundle(p,AUDIT_FILES).values())


def test_audit_zip_traversal_rejected(tmp_path):
    p=tmp_path/'audit.zip'
    with zipfile.ZipFile(p,'w') as z:
        for name in AUDIT_FILES:z.writestr(name,'{}')
        z.writestr('../outside.json','{}')
    with pytest.raises(ValueError,match='Unsafe ZIP'):
        read_bundle(p,AUDIT_FILES)


def test_original_discovery_excludes_the_audits_checkpoint_only_copy(tmp_path):
    original=tmp_path/'original';reference=tmp_path/'audit_output/balora_validation_reference'
    for folder in [original,reference]:
        (folder/'research_fold_01').mkdir(parents=True)
        (folder/'research_fold_01/best.pt').write_bytes(b'fixture')
        for name in ['reproducibility_manifest.json','selected_checkpoint.json','session_state.json']:
            (folder/name).write_text('{}')
    (original/'heldout_evaluation.json').write_text('{}')
    assert locate_original_results(tmp_path)==original


@pytest.mark.parametrize('clean,expected',[(True,55000),(False,54000+43200)])
def test_baseline_resume_uses_cumulative_ledger(tmp_path,clean,expected):
    from evaluation.zero_shot_run import recover_baseline
    prior=tmp_path/'input/prior';prior.mkdir(parents=True)
    (prior/'baseline_session_state.json').write_text(json.dumps({
        'build':BUILD,'checkpoint_sha256':'same','closed_cleanly':clean,
        'accounted_runtime_seconds':55000,'prior_accounted_seconds':54000}))
    (prior/'zero_shot_progress.json').write_text('{"rows":[]}')
    output=tmp_path/'output';output.mkdir()
    context={'prior_accounted_seconds':53000,'reference':{'checkpoint_sha256':'same'}}
    assert recover_baseline(tmp_path/'input',output,context)==expected
    assert (output/'zero_shot_progress.json').is_file()


def test_orchestration_exports_the_comparison_and_closes_ledger(tmp_path,monkeypatch):
    import evaluation.zero_shot_run as runner
    c,model,loader=setup()
    original=evaluate(model,loader,c,tmp_path/'reference_progress.json')
    context=comparison_context(c,original)
    base=tmp_path/'base.pt';base.write_bytes(b'CPU fixture, not real model weights')
    from evaluation.validation_audit import sha256_file
    c['sam2'].update(checkpoint_sha256=sha256_file(base),resolved_commit='fake')
    context['reference']['configuration']=c
    context.update(prior_accounted_seconds=54637.2,policy_selection={'selected_policy':'none'},
                   audit_source='CPU fixture')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU fake integration test')
    monkeypatch.setattr(runner,'BTCVDataset',lambda *a,**k:loader.dataset)
    monkeypatch.setattr(runner,'collate_btcv',collate)
    monkeypatch.setattr(runner,'FrozenSAM2',lambda config,device:FrozenSAM2(config,'cpu',backbone=FakeSAM2(image_size=64)))
    real=runner.run_zero_shot_evaluation
    monkeypatch.setattr(runner,'run_zero_shot_evaluation',lambda model,loader,device,**kw:real(model,loader,'cpu',**kw))
    inputs=tmp_path/'inputs';inputs.mkdir()
    out=tmp_path/'output'
    report=runner.run_comparison(context,tmp_path,out,base,'fake',inputs,
                                  notebook_start=time.time()-10)
    assert report['complete']
    state=json.loads((out/'baseline_session_state.json').read_text())
    assert state['closed_cleanly'] and state['execution_status']=='ZERO_SHOT_COMPARISON_COMPLETE'
    assert state['accounted_runtime_seconds']>=15.1771*3600+10
    with zipfile.ZipFile(out/'BALoRA_ZeroShot_Comparison_Results.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        assert json.loads(z.read(out.name+'/baseline_session_state.json'))['closed_cleanly']
