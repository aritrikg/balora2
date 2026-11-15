import copy
import hashlib
import json
from pathlib import Path
import time
import zipfile
import pytest
import torch
from torch.utils.data import DataLoader,Dataset
from models.deterministic_sam2 import DeterministicSAM2
from models.frozen_sam2 import FrozenSAM2
from models.bayesian_lora import BayesianLoRALinear,DeterministicLoRALinear
from tests.fake_sam2 import FakeSAM2
from tests.test_training_and_refinement_fake import collate
from evaluation.control_inputs import BUILD,protocol_digest
from evaluation.point_estimate import evaluate_point_model
from evaluation.control_run import export_three_way
from training.deterministic_control import (deterministic_objective,control_weights,
    run_fixed_training,plan_control,profile_control)
from training.train import train_one_epoch,_read_checkpoint
from utils.control_session import (LEDGER,ARCHIVE,begin_control_session,
    finish_control_session,export_control_bundle,append_clean_ledger,recover_control)
from utils.runtime_budget import BudgetExpired


def cfg(root):
    c={'sam2':{'model_cfg':'fake','checkpoint':'fake','checkpoint_sha256':'base-test','resolved_commit':'stub'},
       'lora':{'encoder_target_modules':['qkv'],'target_modules':['q_proj','out_proj'],
               'encoder_rank':2,'decoder_rank':2,'alpha':4.,'dropout':.05},
       'data':{'image_size':64,'num_classes':3,'ct_window':[-175.,250.],
               'clip_len':1,'foreground_only':True},
       'loss':{'dice_weight':1.,'bce_weight':1.,'lambda_cal':.1,'lambda_kl':0.,'lambda_rank':0.},
       'training':{'num_epochs':2,'min_epochs':1,'early_stopping_patience':6,'batch_size':1,
                   'lr':.001,'weight_decay':1e-5,'grad_clip_norm':1.,'amp':False,'checkpoint_seconds':300},
       'experiment':{'seed':42,'output_dir':str(root),'split_hash':'test-only',
            'split_case_ids':{'train':['t1','t2'],'validation':['v1','v2'],'test':['x1','x2']}}}
    c['control']={'build':BUILD,'protocol_sha256':protocol_digest(c)}
    return c


class TinyClips(Dataset):
    def __init__(self,cases,slices=2):
        self.case_ids=list(cases)
        self._clip_index=[(cid,z,z+1) for cid in cases for z in range(slices)]
    def __len__(self):return len(self._clip_index)
    def __getitem__(self,index):
        cid,z,_=self._clip_index[index]
        image=torch.rand(1,1,64,64,generator=torch.Generator().manual_seed(2027+index))
        label=torch.zeros(1,64,64,dtype=torch.long)
        label[:,18:40,20:44]=1;label[:,42:51,8:20]=2
        label=torch.nn.functional.one_hot(label,num_classes=3).permute(0,3,1,2).float()
        return {'image':image,'label':label,'case_id':cid,'slice_indices':[z],
                'valid_slice_mask':torch.ones(1,dtype=torch.bool)}


def loader(cases,slices=2):
    return DataLoader(TinyClips(cases,slices),batch_size=1,collate_fn=collate,
                       generator=torch.Generator().manual_seed(123))


def model(c):
    torch.manual_seed(42)
    return DeterministicSAM2(c,'cpu',backbone=FakeSAM2(64))


def eval_model(m,ld,c,path,**kwargs):
    return evaluate_point_model(m,ld,'cpu',allowed_cases=ld.dataset.case_ids,
        seed=700043,progress_path=path,**kwargs)


def test_only_deterministic_adapters_have_trainable_parameters_and_fixed_ranks(tmp_path):
    m=model(cfg(tmp_path))
    assert not hasattr(m,'rank_controller')
    assert not any(isinstance(x,BayesianLoRALinear) for x in m.modules())
    assert len(m.encoder_lora_layers)==1 and len(m.decoder_lora_layers)==2
    assert all(n.endswith(('.lora_A','.lora_B')) for n,p in m.named_parameters() if p.requires_grad)
    assert all(x.rank==2 for x in m.modules() if isinstance(x,DeterministicLoRALinear))
    layer=next(x for x in m.modules() if isinstance(x,DeterministicLoRALinear))
    layer._active_rank_override=1
    with pytest.raises(ValueError,match='fixed rank'):m.assert_control()


def test_control_rejects_an_existing_adapted_backbone(tmp_path):
    c=cfg(tmp_path)
    with pytest.raises(ValueError,match='fresh backbone'):
        DeterministicSAM2(c,'cpu',backbone=model(c).backbone)


def test_fresh_zero_b_factors_preserve_native_prediction(tmp_path):
    c=cfg(tmp_path);torch.manual_seed(9);backbone=FakeSAM2(64)
    fresh=copy.deepcopy(backbone)
    native=FrozenSAM2(c,'cpu',backbone=backbone)
    adapted=DeterministicSAM2(c,'cpu',backbone=fresh)
    image=torch.rand(1,1,64,64);p=torch.tensor([[[22.,25.]]]);y=torch.ones(1,1,dtype=torch.int32)
    with torch.no_grad():
        a=native.decode_one(native.encode_image(image),p,y)
        b=adapted.decode_one(adapted.encode_image(image),p,y)
    assert torch.equal(a,b)


def test_training_changes_both_adapter_scopes_preserves_frozen_backbone(tmp_path):
    c=cfg(tmp_path);m=model(c)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
    progress={}
    loss=train_one_epoch(m,loader(['t1','t2']),opt,control_weights(c),1,'cpu',
        objective_fn=deterministic_objective,progress_state=progress,log_every=100)
    assert loss>0 and progress['updates']==4
    changed=[n for n,p in m.named_parameters() if not torch.equal(before[n],p)]
    assert any('.image_encoder.' in n for n in changed)
    assert any('.sam_mask_decoder.' in n for n in changed)
    assert all(p.requires_grad or torch.equal(before[n],p) for n,p in m.named_parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.backbone.training is False


def test_control_objective_rejects_bayesian_regularization(tmp_path):
    c=cfg(tmp_path);m=model(c);batch=next(iter(loader(['t1'])))
    w=control_weights(c);w.lambda_kl=.001
    with pytest.raises(ValueError,match='zero Bayesian'):
        deterministic_objective(m,batch['image'][:,0],batch['label'][:,0],w,1,False,1)


def test_evaluation_preserves_rng_weights_and_records_real_prompt_coordinates(tmp_path):
    c=cfg(tmp_path);m=model(c);ld=loader(['x1','x2'])
    before={n:p.clone() for n,p in m.state_dict().items()};rng=torch.get_rng_state().clone()
    report=eval_model(m,ld,c,tmp_path/'p.json',surface=True,split='test')
    assert report['complete'] and report['protocol']['n_scored_slice_organ_predictions']==8
    assert not report['protocol']['epistemic_uncertainty_estimated']
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p,before[n]) for n,p in m.state_dict().items())
    rows=json.loads((tmp_path/'p.json').read_text())['rows']
    from evaluation.evaluate import _prediction_seed
    from utils.prompting import sample_foreground_points
    for r in rows:
        ix=next(i for i,v in enumerate(ld.dataset._clip_index) if v[:2]==(r['case_id'],r['slice_index']))
        target=ld.dataset[ix]['label'][0,r['organ_class']].unsqueeze(0)
        torch.manual_seed(_prediction_seed(700043,r['case_id'],r['slice_index'],r['organ_class']))
        assert r['prompt_xy']==sample_foreground_points(target)[0,0].tolist()


def test_completed_evaluation_resume_does_not_repeat_decoding(tmp_path,monkeypatch):
    c=cfg(tmp_path);m=model(c);ld=loader(['x1']);path=tmp_path/'p.json'
    first=eval_model(m,ld,c,path)
    def forbidden(*a,**k):raise AssertionError('Repeated decoding')
    monkeypatch.setattr(m,'decode_one',forbidden)
    assert eval_model(m,ld,c,path)['segmentation']==first['segmentation']


def test_partial_evaluation_and_changed_adapter_identity(tmp_path):
    c=cfg(tmp_path);m=model(c);ld=loader(['x1']);path=tmp_path/'p.json'
    assert not eval_model(m,ld,c,path,deadline=time.time()-1)['complete']
    assert eval_model(m,ld,c,path)['complete']
    with torch.no_grad():next(p for p in m.parameters() if p.requires_grad).add_(.1)
    with pytest.raises(ValueError,match='another model'):
        eval_model(m,ld,c,path)


def test_out_of_split_evaluation_and_training_are_blocked(tmp_path):
    c=cfg(tmp_path);m=model(c)
    with pytest.raises(ValueError,match='declared split'):
        evaluate_point_model(m,loader(['x1']),'cpu',allowed_cases=['v1'],seed=1,progress_path=tmp_path/'p.json')
    with pytest.raises(ValueError,match='frozen split'):
        run_fixed_training(c,loader(['x1']),loader(['v1','v2']),'cpu',model=m)


def test_partial_epoch_resume_reproduces_continuous_updates_and_scheduler(tmp_path,monkeypatch):
    from training import deterministic_control as module
    a=cfg(tmp_path/'full');full,fs=run_fixed_training(a,loader(['t1','t2']),loader(['v1','v2']),'cpu',model=model(a))
    b=cfg(tmp_path/'resume');real=module.train_one_epoch
    def pause(*args,**kwargs):
        callback=kwargs['progress_callback']
        def stop(state):
            callback(state)
            if state['next_batch']==2:raise BudgetExpired('intentional CPU test pause')
        kwargs['progress_callback']=stop
        return real(*args,**kwargs)
    monkeypatch.setattr(module,'train_one_epoch',pause)
    _,partial=run_fixed_training(b,loader(['t1','t2']),loader(['v1','v2']),'cpu',model=model(b))
    assert not partial['protocol_complete'] and partial['optimizer_updates']==2
    checkpoint=_read_checkpoint(Path(b['experiment']['output_dir'])/'last.pt')
    assert checkpoint['next_batch']==2 and checkpoint['next_epoch']==0
    monkeypatch.setattr(module,'train_one_epoch',real)
    resumed,rs=run_fixed_training(b,loader(['t1','t2']),loader(['v1','v2']),'cpu',model=model(b))
    assert rs['protocol_complete'] and rs['optimizer_updates']==fs['optimizer_updates']==8
    assert all(torch.equal(p,dict(resumed.named_parameters())[n]) for n,p in full.named_parameters())
    assert [(r['train_loss'],r['val_dice'],r['lr_after_epoch']) for r in rs['history']]==[(r['train_loss'],r['val_dice'],r['lr_after_epoch']) for r in fs['history']]


def test_validation_pause_does_not_repeat_finished_epoch_training(tmp_path,monkeypatch):
    from training import deterministic_control as module
    c=cfg(tmp_path);real=module.evaluate_point_model
    def pause(*args,**kwargs):
        kwargs['deadline']=time.time()-1
        return real(*args,**kwargs)
    monkeypatch.setattr(module,'evaluate_point_model',pause)
    _,s=run_fixed_training(c,loader(['t1','t2']),loader(['v1','v2']),'cpu',model=model(c))
    assert not s['protocol_complete'] and s['optimizer_updates']==4
    monkeypatch.setattr(module,'evaluate_point_model',real)
    _,done=run_fixed_training(c,loader(['t1','t2']),loader(['v1','v2']),'cpu',model=model(c))
    assert done['protocol_complete'] and done['optimizer_updates']==8


def test_minimum_epochs_and_saved_completion_are_honored(tmp_path,monkeypatch):
    from training import deterministic_control as module
    c=cfg(tmp_path);c['training'].update(num_epochs=20,min_epochs=8)
    c['control']['protocol_sha256']=protocol_digest(c)
    def constant(*a,**k):return {'complete':True,'segmentation':{'dice':{'mean':.5}},'calibration':{'ece':{'mean':.1}}}
    monkeypatch.setattr(module,'evaluate_point_model',constant)
    _,done=run_fixed_training(c,loader(['t1','t2'],slices=1),loader(['v1','v2'],slices=1),'cpu',model=model(c))
    assert done['completed_epochs']==8 and done['best_epoch']==0 and done['stop_reason']=='EARLY_STOPPING_COMPLETED'
    def forbidden(*a,**k):raise AssertionError('Training repeated after completion')
    monkeypatch.setattr(module,'train_one_epoch',forbidden)
    _,again=run_fixed_training(c,loader(['t1','t2'],slices=1),loader(['v1','v2'],slices=1),'cpu',model=model(c))
    assert again['completed_epochs']==8 and again['protocol_complete']


def test_resume_rejects_changed_training_protocol(tmp_path):
    c=cfg(tmp_path)
    run_fixed_training(c,loader(['t1','t2']),loader(['v1','v2']),'cpu',model=model(c),deadline=time.time()-1)
    c['lora']['alpha']=5.;c['control']['protocol_sha256']=protocol_digest(c)
    with pytest.raises(ValueError,match='matching deterministic'):
        run_fixed_training(c,loader(['t1','t2']),loader(['v1','v2']),'cpu',model=model(c))


def test_profile_is_disposable_updates_both_scopes_and_records_all_train_cases(tmp_path):
    c=cfg(tmp_path)
    p=profile_control(c,loader(['t1','t2']),loader(['v1','v2']),'cpu',deadline=time.time()+60,model=model(c))
    assert p['complete'] and p['profile_model_discarded_before_research_training']
    assert p['training_sample_cases']==['t1','t2']
    assert p['changed_trainable_tensors']>0 and p['profile_updates']==4
    assert not (tmp_path/'last.pt').exists()


def test_full_twenty_epoch_budget_gate_never_relabels_a_shortened_target():
    now=time.time()
    profile={'training_p75_seconds_per_slice':2.,'train_foreground_slices':1347,
        'validation_sample_seconds':30.,'validation_sample_slices':30,'validation_foreground_slices':371}
    state={'work_deadline_unix':now+11*3600,'cumulative_work_deadline_unix':now+14*3600}
    plan=plan_control(profile,state,1000,now=now)
    assert not plan['requested_protocol_estimated_to_fit'] and plan['requested_total_epochs']==20
    assert plan['estimated_final_evaluation_reserve_seconds']>=1800
    assert plan_control(profile,state,1000,now=now,completed_epochs=17,training_complete=True)['remaining_target_epochs']==0


def test_clean_ledger_retries_do_not_double_count_and_keep_session_deadline(tmp_path):
    root=tmp_path/'out';inp=tmp_path/'input';inp.mkdir();start=1000000.
    first=begin_control_session(inp,root,'identity',15*3600,notebook_start=start,
        process_token='kernel',now=start,minimum_prior_hours=0)
    finish_control_session(root,first,'PARTIAL',now=start+1800)
    again=begin_control_session(inp,root,'identity',15*3600,notebook_start=start,
        process_token='kernel',now=start+2000,minimum_prior_hours=0)
    assert again['prior_accounted_seconds']==15.5*3600
    assert again['accounting_start_unix']==start+1800
    assert again['session_start_unix']==start
    closed=finish_control_session(root,again,'PARTIAL',now=start+3600)
    assert closed['accounted_runtime_seconds']==16*3600


def test_active_same_kernel_recovery_does_not_charge_twelve_hours(tmp_path):
    root=tmp_path/'out';inp=tmp_path/'input';inp.mkdir();start=1000000.
    begin_control_session(inp,root,'id',15*3600,notebook_start=start,process_token='same',now=start,minimum_prior_hours=0)
    s=begin_control_session(inp,root,'id',15*3600,notebook_start=start,process_token='same',now=start+1800,minimum_prior_hours=0)
    assert s['prior_accounted_seconds']==15*3600 and s['accounted_runtime_seconds']==15.5*3600


def test_foreign_unclosed_session_is_conservatively_charged(tmp_path):
    root=tmp_path/'out';inp=tmp_path/'input';inp.mkdir();start=1000000.
    begin_control_session(inp,root,'id',15*3600,notebook_start=start,process_token='old',now=start,minimum_prior_hours=0)
    s=begin_control_session(inp,root,'id',15*3600,notebook_start=start+12*3600,
        process_token='new',now=start+12*3600,minimum_prior_hours=0)
    assert s['prior_accounted_seconds']==27*3600
    assert s['accounting_reason']=='unclean_resume_charged_12h'


def test_portable_bundle_recovery_contains_weights_but_not_cache(tmp_path):
    root=tmp_path/'out';inp=tmp_path/'input';inp.mkdir();start=1000000.
    s=begin_control_session(inp,root,'id',15*3600,notebook_start=start,process_token='old',now=start,minimum_prior_hours=0)
    (root/'last.pt').write_bytes(b'control checkpoint')
    (root/'raw_volume_cache').mkdir();(root/'raw_volume_cache/raw.npy').write_bytes(b'raw CT')
    archive=export_control_bundle(root)
    finish_control_session(root,s,'PARTIAL',now=start+1800);append_clean_ledger(archive,root)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert not any('raw_volume_cache' in n for n in z.namelist())
    restored=tmp_path/'restored'
    recovered=recover_control(inp,restored,'id',str(archive))
    assert recovered['closed_cleanly'] and (restored/'last.pt').read_bytes()==b'control checkpoint'


def test_mismatched_recovery_and_zip_traversal_are_rejected(tmp_path):
    root=tmp_path/'out';inp=tmp_path/'input';inp.mkdir();start=1000000.
    begin_control_session(inp,root,'id',15*3600,notebook_start=start,process_token='old',now=start,minimum_prior_hours=0)
    with pytest.raises(ValueError,match='different protocol'):
        recover_control(inp,root,'wrong')
    archive=inp/ARCHIVE
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr(LEDGER,json.dumps({'build':BUILD,'protocol_sha256':'id'}))
        z.writestr('../outside.json','{}')
    with pytest.raises(ValueError,match='Unsafe'):
        recover_control(inp,tmp_path/'fresh','id',str(archive))


def comparison_context(c,report):
    return {'reference':{'checkpoint_sha256':'original-adapter','splits':c['experiment']['split_case_ids']},
        'zero_shot':copy.deepcopy(report),'original_balora':copy.deepcopy(report),'seed':700043,
        'expected_counts':{(o,cid):r['n_slices'] for o,vols in report['per_volume_segmentation'].items() for cid,r in vols.items()}}


def test_three_way_comparison_requires_complete_training_and_exact_coverage(tmp_path):
    c=cfg(tmp_path);m=model(c);report=eval_model(m,loader(['x1','x2']),c,tmp_path/'test.json',surface=True,split='test')
    context=comparison_context(c,report);(tmp_path/'best.pt').write_bytes(b'test control')
    done={'protocol_complete':True}
    result=export_three_way(context,report,done,c,tmp_path)
    assert result['complete'] and len(result['methods'])==3 and result['n_predictions_per_method']==8
    with pytest.raises(ValueError,match='Partial training'):
        export_three_way(context,report,{'protocol_complete':False},c,tmp_path)
    context['expected_counts'][('spleen','missing')]=1
    with pytest.raises(ValueError,match='identical'):
        export_three_way(context,report,done,c,tmp_path)


def test_official_runtime_factory_constructs_control_without_bayesian_builder(tmp_path,monkeypatch):
    import sys,types
    c=cfg(tmp_path);calls=[]
    package=types.ModuleType('sam2');package.__path__=[]
    factory=types.ModuleType('sam2.build_sam')
    def build(model_cfg,checkpoint,device,mode):
        calls.append((model_cfg,checkpoint,device,mode));return FakeSAM2(64)
    factory.build_sam2=build
    monkeypatch.setitem(sys.modules,'sam2',package)
    monkeypatch.setitem(sys.modules,'sam2.build_sam',factory)
    m=DeterministicSAM2(c,'cpu')
    assert calls==[('fake','fake','cpu','train')]
    assert not m.test_backbone_injected and len(m.decoder_lora_layers)==2
    assert not any('mu_' in n or 'log_sigma' in n for n,_ in m.named_parameters())


def test_full_orchestration_cpu_boundary_routes_real_training_and_evaluation(tmp_path,monkeypatch):
    from evaluation import control_run as module
    c=cfg(tmp_path/'out');root=Path(c['experiment']['output_dir']);root.mkdir(parents=True)
    checkpoint=tmp_path/'official-base.pt';checkpoint.write_bytes(b'CPU boundary fixture, never a research checkpoint')
    c['sam2']['checkpoint']=str(checkpoint);c['sam2']['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    c['control']['protocol_sha256']=protocol_digest(c)
    reference_report=eval_model(model(c),loader(['x1','x2']),c,tmp_path/'ref.json',surface=True,split='test')
    reference_report['elapsed_seconds_this_session']=1000
    context=comparison_context(c,reference_report)
    inp=tmp_path/'input';inp.mkdir();requested=[]
    def cpu_loader(config,split):
        if split=='test':
            assert json.loads((root/'training_status.json').read_text())['protocol_complete']
        requested.append(split)
        return loader(config['experiment']['split_case_ids'][split])
    real_train=module.run_fixed_training;real_eval=module.evaluate_point_model
    def cpu_train(config,train,val,device,**kwargs):
        return real_train(config,train,val,'cpu',model=model(config),**kwargs)
    def cpu_profile(config,train,val,device,**kwargs):
        return profile_control(config,train,val,'cpu',model=model(config),**kwargs)
    def cpu_eval(m,ld,device,**kwargs):return real_eval(m,ld,'cpu',**kwargs)
    monkeypatch.setattr('evaluation.full_figures.export_full_figures',lambda *a,**k:None)
    monkeypatch.setattr(module,'make_loader',cpu_loader)
    monkeypatch.setattr(module,'run_fixed_training',cpu_train)
    monkeypatch.setattr(module,'profile_control',cpu_profile)
    monkeypatch.setattr(module,'evaluate_point_model',cpu_eval)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'CPU routed orchestration fixture')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    state=begin_control_session(inp,root,c['control']['protocol_sha256'],15.4852*3600,
        notebook_start=time.time(),process_token='test-kernel')
    result=module.run_control(context,c,state)
    assert result['complete'] and requested==['train','validation','test']
    assert (root/ARCHIVE).is_file()
    with zipfile.ZipFile(root/ARCHIVE) as z:
        assert z.testzip() is None
        ledger=json.loads(z.read(root.name+'/'+LEDGER))
        assert ledger['closed_cleanly'] and ledger['execution_status']=='DETERMINISTIC_CONTROL_COMPLETE'
        assert any(n.endswith('/last.pt') for n in z.namelist())
        assert any(n.endswith('/best.pt') for n in z.namelist())
        assert not any('raw_volume_cache' in n for n in z.namelist())
    def forbidden(*a,**k):raise AssertionError('Repeated complete control execution')
    monkeypatch.setattr(module,'make_loader',forbidden)
    again=begin_control_session(inp,root,c['control']['protocol_sha256'],15.4852*3600,
        notebook_start=state['session_start_unix'],process_token='test-kernel')
    assert module.run_control(context,c,again)['methods']==result['methods']


def test_partial_training_never_opens_test_data_or_exports_complete_comparison(tmp_path,monkeypatch):
    from evaluation import control_run as module
    c=cfg(tmp_path/'out');root=Path(c['experiment']['output_dir']);root.mkdir(parents=True)
    p=tmp_path/'base.pt';p.write_bytes(b'CPU-only base fixture')
    c['sam2']['checkpoint']=str(p);c['sam2']['checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    c['control']['protocol_sha256']=protocol_digest(c)
    reference=eval_model(model(c),loader(['x1','x2']),c,tmp_path/'r.json',surface=True,split='test')
    context=comparison_context(c,reference)
    def guarded_loader(config,split):
        assert split!='test','Test data opened before training completion'
        return loader(config['experiment']['split_case_ids'][split])
    real=module.run_fixed_training
    def paused(config,train,val,device,**kwargs):
        return real(config,train,val,'cpu',model=model(config),deadline=time.time()-1)
    def profile(*a,**k):return {'training_p75_seconds_per_slice':.1,'train_foreground_slices':4,
        'validation_sample_seconds':1.,'validation_sample_slices':4,'validation_foreground_slices':4}
    monkeypatch.setattr(module,'make_loader',guarded_loader)
    monkeypatch.setattr(module,'run_fixed_training',paused)
    monkeypatch.setattr(module,'profile_control',profile)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    inp=tmp_path/'input';inp.mkdir()
    state=begin_control_session(inp,root,c['control']['protocol_sha256'],15.4852*3600,
        notebook_start=time.time(),process_token='test')
    result=module.run_control(context,c,state)
    assert not result['complete'] and result['status']=='PARTIAL_CONTROL_TRAINING_RESUME_REQUIRED'
    assert not (root/'method_comparison.json').exists()
    assert not (root/'control_test_evaluation.json').exists()
    assert json.loads((root/LEDGER).read_text())['execution_status']==result['status']
