"""Lifecycle regressions; tiny fake SAM2 tests are not research GPU results."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader
from tests.test_sam2_wrapper_fake import build_fake
from tests.test_training_and_refinement_fake import OneClip,collate
from tests.test_evaluation_export import TinyEvalSet
from datasets.btcv_dataset import BTCVDataset,collate_btcv
from evaluation.evaluate import run_evaluation
from evaluation.metrics import negative_log_likelihood,hausdorff_distance_95
from utils.checkpoint import load_checkpoint,save_checkpoint
from utils.runtime_budget import plan_runtime
from training import train as tr


def tiny_config(tmp_path, epochs=2):
    return {'experiment':{'device':'cpu','output_dir':str(tmp_path),'seed':42,'split_hash':'tiny-split'},
        'data':{'image_size':64,'num_classes':2,'ct_window':[-175,250]},
        'sam2':{'model_cfg':'fake','checkpoint':'fake'},
        'lora':{'target_modules':['q_proj','out_proj'],'encoder_target_modules':['qkv'],
                'rank_bank':[1,2],'alpha':2.0,'dropout':0.0},
        'bayesian':{'k_train':2,'k_infer':2},'loss':{'lambda_kl':0.00001},
        'training':{'batch_size':1,'num_epochs':epochs,'lr':0.001,'amp':False,'min_epochs':1,
            'early_stopping_patience':10,'checkpoint_seconds':0},'logging':{'log_every_n_steps':1000}}


def patch_tiny(monkeypatch):
    def factory(cfg,device):
        m=build_fake();m._run_config=cfg;return m
    def loaders(cfg):
        return DataLoader(OneClip(),batch_size=1,collate_fn=collate),DataLoader(OneClip(),batch_size=1,collate_fn=collate)
    monkeypatch.setattr(tr,'build_model',factory)
    monkeypatch.setattr(tr,'build_dataloaders',loaders)


def test_training_saves_best_last_and_resume_restores_best(tmp_path,monkeypatch):
    patch_tiny(monkeypatch)
    cfg=tiny_config(tmp_path/'first')
    out=tr.run_training(cfg)
    assert out['protocol_complete'] and out['completed_epochs']==2
    last=tr._read_checkpoint(out['last_checkpoint'])
    assert last['checkpoint_format']=='adapters_v1'
    assert last['best_model_state_dict']
    assert all('base_layer' not in k for k in last['model_state_dict'])
    cfg2=copy.deepcopy(cfg);cfg2['experiment']['output_dir']=str(tmp_path/'resume')
    resumed=tr.run_training(cfg2,resume_from=out['last_checkpoint'])
    assert resumed['best_checkpoint'] and resumed['completed_epochs']==2
    best=tr._read_checkpoint(resumed['best_checkpoint'])
    assert all(torch.equal(v,last['best_model_state_dict'][k]) for k,v in best['model_state_dict'].items())


def test_budget_pause_is_not_success_and_resumes(tmp_path,monkeypatch):
    patch_tiny(monkeypatch)
    cfg=tiny_config(tmp_path/'paused',epochs=1)
    cfg['runtime']={'training_deadline_unix':time.time()-1}
    out=tr.run_training(cfg)
    assert out['stop_reason']=='TIME_BUDGET_PAUSED' and not out['protocol_complete']
    assert out['completed_epochs']==0 and Path(out['last_checkpoint']).exists()
    cfg['runtime']={}
    resumed=tr.run_training(cfg,resume_from=out['last_checkpoint'])
    assert resumed['protocol_complete'] and resumed['completed_epochs']==1


def test_eval_fixed_seed_restores_rng_and_progress_resumes(tmp_path):
    torch.manual_seed(123);m=build_fake()
    # Fixed inputs; TinyEvalSet otherwise generates a different random image.
    sample=TinyEvalSet()[0]
    loader=DataLoader([sample],batch_size=1,collate_fn=collate_btcv)
    before=torch.get_rng_state().clone()
    r=run_evaluation(m,loader,'cpu',2,compute_surface_distances=False,seed=77,progress_path=tmp_path/'progress.json')
    assert torch.equal(before,torch.get_rng_state())
    r2=run_evaluation(m,loader,'cpu',2,compute_surface_distances=False,seed=77,progress_path=tmp_path/'progress.json')
    assert r['segmentation']==r2['segmentation']
    assert r2['protocol']['complete']
    paused=run_evaluation(m,loader,'cpu',2,compute_surface_distances=False,deadline=time.time()-1)
    assert not paused['protocol']['complete']
    with torch.no_grad(): next(p for p in m.parameters() if p.requires_grad).add_(1)
    with pytest.raises(ValueError,match='different weights'):
        run_evaluation(m,loader,'cpu',2,compute_surface_distances=False,seed=77,progress_path=tmp_path/'progress.json')


def test_metric_probability_endpoints_and_nested_surfaces():
    p=torch.tensor([0.,1.,0.,1.]);y=torch.tensor([0.,1.,1.,0.])
    assert np.isfinite(negative_log_likelihood(p,y))
    outer=torch.zeros(30,30,dtype=torch.bool);outer[2:28,2:28]=True
    inner=torch.zeros_like(outer);inner[10:20,10:20]=True
    # Inner boundaries lie inside the outer mask; their distance to the
    # outer SURFACE must be positive (not zero distance to the filled mask).
    from scipy.ndimage import binary_erosion
    from scipy.spatial.distance import cdist
    a=np.argwhere(outer.numpy() & ~binary_erosion(outer.numpy()))
    b=np.argwhere(inner.numpy() & ~binary_erosion(inner.numpy()))
    d=cdist(a,b);expected=np.percentile(np.r_[d.min(1),d.min(0)],95)
    assert hausdorff_distance_95(outer,inner)==pytest.approx(expected)


def test_cached_dataset_matches_uncached_and_filters_only_background(tmp_path,monkeypatch):
    monkeypatch.delenv("BALORA_CACHE_DIR",raising=False)
    import nibabel as nib
    root=tmp_path/'data';(root/'img').mkdir(parents=True);(root/'label').mkdir()
    x=np.linspace(-300,400,8*8*4,dtype=np.float32).reshape(8,8,4)
    y=np.zeros_like(x);y[2:5,3:6,1:3]=1
    nib.save(nib.Nifti1Image(x,np.eye(4)),root/'img/img0001.nii.gz')
    nib.save(nib.Nifti1Image(y,np.eye(4)),root/'label/label0001.nii.gz')
    split=tmp_path/'split.json';split.write_text(json.dumps({'cases':['img0001']}))
    a=BTCVDataset(str(root),str(split),2,image_size=16)
    b=BTCVDataset(str(root),str(split),2,image_size=16,cache_dir=str(tmp_path/'cache'),foreground_only=True)
    assert len(a)==4 and len(b)==2 and b.background_clips_removed==2
    assert b._clip_index==[('img0001',1,2),('img0001',2,3)]
    assert torch.equal(a[1].image,b[0].image) and torch.equal(a[1].label,b[0].label)
    uncached=BTCVDataset(str(root),str(split),2,image_size=16,foreground_only=True)
    assert uncached._clip_index==b._clip_index


def test_runtime_gate_preserves_requested_epochs():
    p=plan_runtime(3600,600,elapsed_seconds=1000,budget_hours=10,requested_epochs=100)
    assert p['requested_total_epochs']==100
    assert not p['requested_protocol_estimated_to_fit']
    assert p['estimated_feasible_additional_epochs']<100


def test_background_batch_does_not_call_encoder():
    from losses.losses import LossWeights
    m=build_fake()
    def fail(*args,**kwargs): raise AssertionError('Background clip reached encoder')
    m.encode_image=fail
    batch=next(iter(DataLoader(OneClip(),batch_size=1,collate_fn=collate)))
    batch['label'][:,:,1]=0
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=0.001)
    with pytest.raises(RuntimeError,match='zero optimizer updates'):
        tr.train_one_epoch(m,[batch],opt,LossWeights(),2,'cpu')


def test_legacy_checkpoint_is_revalidated_before_continuation(tmp_path,monkeypatch):
    patch_tiny(monkeypatch)
    cfg=tiny_config(tmp_path/'old',epochs=1)
    out=tr.run_training(cfg)
    # Mimic the old full-weight checkpoint: no fixed-prompt selection metadata.
    payload=tr._read_checkpoint(out['best_checkpoint'])
    payload.pop('selection_seed_policy',None)
    payload['best_val_dice']=0.999999
    legacy=tmp_path/'legacy/best.pt';legacy.parent.mkdir();torch.save(payload,legacy)
    cfg['training']['num_epochs']=2
    cfg['experiment']['output_dir']=str(tmp_path/'continued')
    resumed=tr.run_training(cfg,resume_from=str(legacy))
    assert resumed['protocol_complete']
    revalidation=json.loads((tmp_path/'continued/legacy_checkpoint_revalidation.json').read_text())
    assert revalidation['protocol']['complete']
    assert resumed['best_val_dice']<0.999999
    assert resumed['continuation_schedule']


def test_resume_rejects_changed_seed_for_partial_epoch(tmp_path):
    a=tiny_config(tmp_path);b=copy.deepcopy(a);b['experiment']['seed']=99
    with pytest.raises(ValueError,match='experiment.seed'):
        tr._assert_resume_identity(a,b)


def test_evaluation_preserves_same_checkpoint_unrefined_comparison():
    from models.refinement import RefinementConfig
    m=build_fake();sample=TinyEvalSet()[0]
    loader=DataLoader([sample],batch_size=1,collate_fn=collate_btcv)
    result=run_evaluation(m,loader,'cpu',2,refinement_config=RefinementConfig(max_iters=0),
        compute_surface_distances=False,include_unrefined=True)
    assert result['segmentation']==result['same_checkpoint_no_refinement']['segmentation']


def test_partial_epoch_resumes_without_repeating_optimizer_update(tmp_path,monkeypatch):
    patch_tiny(monkeypatch)
    # Two fixed samples. Pause immediately after saving the first real update.
    samples=[TinyEvalSet()[0],TinyEvalSet()[0]]
    loader=DataLoader(samples,batch_size=1,collate_fn=collate_btcv)
    monkeypatch.setattr(tr,'build_dataloaders',lambda cfg:(loader,loader))
    cfg=tiny_config(tmp_path/'paused',epochs=1)
    orig=tr.save_checkpoint; paused=[False]
    def save_then_pause(*args,**kwargs):
        orig(*args,**kwargs)
        state=kwargs.get('extra',{})
        if state.get('next_batch')==1 and not paused[0]:
            paused[0]=True
            raise tr.BudgetExpired('Test interruption after one consistent update')
    monkeypatch.setattr(tr,'save_checkpoint',save_then_pause)
    out=tr.run_training(cfg)
    assert not out['protocol_complete']
    p=tr._read_checkpoint(out['last_checkpoint'])
    assert p['next_batch']==1 and p['partial_progress']['updates']==1
    monkeypatch.setattr(tr,'save_checkpoint',orig)
    resumed=tr.run_training(cfg,resume_from=out['last_checkpoint'])
    assert resumed['protocol_complete']
    last=tr._read_checkpoint(resumed['last_checkpoint'])
    assert last['next_batch']==0
    # Every trained tensor with Adam state has exactly two optimizer updates.
    assert all(int(s['step'])==2 for s in last['optimizer_state_dict']['state'].values() if 'step' in s)


def test_genuine_fp32_nonfinite_gradients_fail_explicitly():
    from losses.losses import LossWeights
    m=build_fake()
    p=next(p for p in m.backbone.image_encoder.parameters() if p.requires_grad)
    hook=p.register_hook(lambda g:torch.full_like(g,float('inf')))
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=0.001)
    loader=DataLoader(OneClip(),batch_size=1,collate_fn=collate)
    with pytest.raises(FloatingPointError,match='Persistent non-finite gradients'):
        tr.train_one_epoch(m,loader,opt,LossWeights(),2,'cpu')
    hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real CUDA AMP execution is required')
def test_cuda_amp_overflow_retries_same_batch_without_losing_update():
    from losses.losses import LossWeights
    m=build_fake().cuda();calls=[0]
    p=next(p for p in m.backbone.image_encoder.parameters() if p.requires_grad)
    def overflow_once(g):
        calls[0]+=1
        return torch.full_like(g,float('inf')) if calls[0]==1 else g
    hook=p.register_hook(overflow_once)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=0.001)
    loader=DataLoader(OneClip(),batch_size=1,collate_fn=collate)
    before=m.backbone.sam_mask_decoder.q_proj.mu_B.detach().clone()
    value=tr.train_one_epoch(m,loader,opt,LossWeights(),2,'cuda',amp=True)
    hook.remove()
    assert np.isfinite(value)
    assert m.last_training_stats['updates']==1
    assert m.last_training_stats['amp_overflow_retries']==1
    assert not torch.equal(before,m.backbone.sam_mask_decoder.q_proj.mu_B)


def _observed_t4_plan(**overrides):
    observed=json.loads((Path(__file__).parent/'fixtures/runtime_t4_20261024.json').read_text())
    kwargs=dict(train_seconds=observed['train_p75_seconds_per_foreground_slice']*observed['train_foreground_slices'],
        validation_seconds=observed['validation_sample_seconds']/observed['validation_sample_size']*observed['validation_foreground_slices'],
        elapsed_seconds=observed['elapsed_before_training_seconds'],budget_hours=29.5,
        final_hours=4.0,safety_hours=0.5,requested_epochs=20,min_epochs=8,
        final_seconds_estimate=observed['estimated_final_seconds_conservative_proxy'])
    kwargs.update(overrides)
    return observed,plan_runtime(**kwargs)


def test_joint_budget_accepts_supplied_t4_measurements_without_shortening_protocol():
    observed,p=_observed_t4_plan()
    # The rejected notebook's training plan already fitted: ONLY its rigid
    # four-hour final-stage gate blocked this otherwise feasible run.
    assert observed['plan']['requested_protocol_estimated_to_fit']
    assert observed['estimated_final_seconds_conservative_proxy']>4*3600
    assert p['requested_protocol_estimated_to_fit']
    assert p['requested_total_epochs']==20
    assert p['reserved_final_hours']==4.5
    assert p['estimated_total_accounted_hours']<29.5
    assert p['training_allocation_hours_from_start']==24.5
    assert p['rejection_reasons']==[]


def test_joint_budget_still_rejects_a_genuinely_oversized_combined_run():
    _,p=_observed_t4_plan(final_seconds_estimate=20*3600)
    assert not p['requested_protocol_estimated_to_fit']
    assert p['requested_total_epochs']==20
    assert p['minimum_protocol_estimated_to_fit']  # must NOT silently switch to eight epochs
    assert p['rejection_reasons']==['insufficient_time_for_requested_training_and_final_evaluation']


def test_completed_training_does_not_hide_an_infeasible_final_stage():
    _,p=_observed_t4_plan(already_completed=20,elapsed_seconds=28*3600)
    assert not p['requested_protocol_estimated_to_fit']
    assert p['estimated_feasible_additional_epochs']==0
    assert not p['final_evaluation_window_estimated_to_fit']
    assert p['rejection_reasons']==['insufficient_total_time_for_final_evaluation']


def test_joint_budget_rounds_final_reserve_up_and_preserves_safety_margin():
    _,a=_observed_t4_plan(final_seconds_estimate=4*3600)
    _,b=_observed_t4_plan(final_seconds_estimate=4*3600+1)
    assert a['reserved_final_hours']==4.0
    assert b['reserved_final_hours']==4.5
    assert b['training_allocation_hours_from_start']+b['reserved_final_hours']+b['safety_hours']==29.5


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.0])
def test_joint_budget_rejects_invalid_final_estimates(bad):
    with pytest.raises(ValueError,match='Final evaluation estimate'):
        _observed_t4_plan(final_seconds_estimate=bad)


def test_prior_compute_and_resume_epochs_are_accounted_jointly():
    _,p=_observed_t4_plan(elapsed_seconds=5*3600+728.1,already_completed=5)
    assert p['requested_protocol_estimated_to_fit']
    assert p['requested_total_epochs']==20 and p['completed_epochs_on_resume']==5
    assert p['estimated_total_accounted_hours']<29.5
