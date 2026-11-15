"""Fresh deterministic LoRA training, fixed validation selection, exact resume."""
from __future__ import annotations
import copy
import gc
import json
import math
from pathlib import Path
import statistics
import time
import torch
from torch.utils.data import DataLoader,Subset
from evaluation.control_inputs import BUILD,protocol_digest
from evaluation.point_estimate import evaluate_point_model
from losses.losses import LossWeights,compute_total_loss
from models.deterministic_sam2 import DeterministicSAM2
from training.train import train_one_epoch,_make_grad_scaler,_adapter_snapshot,_read_checkpoint
from utils.checkpoint import save_checkpoint,load_checkpoint
from utils.prompting import sample_foreground_points
from utils.runtime_budget import BudgetExpired,check_deadline,write_json
from utils.seed import set_seed

RUNTIME_REVISION='measured_activation_policy_v1_3'


def deterministic_objective(model,images,labels,weights,k_train,encoder_amp,kl_count):
    if k_train!=1 or weights.lambda_kl or weights.lambda_rank:
        raise ValueError('Deterministic control requires one prediction and zero Bayesian/rank penalties')
    embedding=model.encode_image(images)
    losses=[];calibration=[];zero=images.new_zeros(())
    for class_id in range(1,labels.shape[1]):
        present=labels[:,class_id].flatten(1).any(1)
        if not bool(present.any()):continue
        target=labels[present,class_id].unsqueeze(1)
        points=sample_foreground_points(target[:,0])
        point_labels=torch.ones(points.shape[:2],dtype=torch.int32,device=images.device)
        logits=model.decode_one(embedding[present],points,point_labels).float()
        # Match the original probability clamp and segmentation/calibration
        # formulation, with one point-estimate prediction instead of MC means.
        mean_logits=torch.logit(logits.sigmoid().clamp(1e-6,1-1e-6))
        loss=compute_total_loss(mean_logits,target.float(),zero,zero,weights)
        if not torch.isfinite(loss.total):raise FloatingPointError('Non-finite deterministic objective')
        losses.append(loss.seg);calibration.append(loss.calibration)
    if not losses:raise ValueError('No foreground reached the deterministic objective')
    seg=torch.stack(losses).mean();cal=torch.stack(calibration).mean()
    return seg+weights.lambda_cal*cal,{'seg':seg,'cal':cal,'kl':zero,'rank':zero}


def control_weights(cfg):
    return LossWeights(**{k:cfg['loss'][k] for k in ('dice_weight','bce_weight','lambda_kl','lambda_cal','lambda_rank')})


def run_fixed_training(cfg,train_loader,val_loader,device,*,deadline=None,model=None):
    train_cases=set(cfg['experiment']['split_case_ids']['train'])
    validation_cases=set(cfg['experiment']['split_case_ids']['validation'])
    if set(train_loader.dataset.case_ids)!=train_cases or set(val_loader.dataset.case_ids)!=validation_cases or train_cases & validation_cases:
        raise ValueError('Control training/validation loader identities do not match the frozen split')
    if protocol_digest(cfg)!=cfg['control']['protocol_sha256']:
        raise ValueError('Control protocol changed after it was frozen')
    root=Path(cfg['experiment']['output_dir']);root.mkdir(parents=True,exist_ok=True)
    seed=int(cfg['experiment']['seed'])
    if model is None:
        set_seed(seed)
        model=DeterministicSAM2(cfg,device)
    model.assert_control()
    trainable=[p for p in model.parameters() if p.requires_grad]
    tr=cfg['training'];target=int(tr['num_epochs'])
    optimizer=torch.optim.AdamW(trainable,lr=tr['lr'],weight_decay=tr['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=target)
    scaler=_make_grad_scaler(bool(tr['amp'] and str(device).startswith('cuda')))
    history=[];best_dice=-float('inf');best_epoch=-1;best_state=None;patience=0
    next_epoch=0;progress={};complete=False;stop_reason='BUDGET_PAUSED'
    last=root/'last.pt';best=root/'best.pt';weights=control_weights(cfg)
    if last.is_file():
        payload=_read_checkpoint(last)
        if payload.get('control_protocol_sha256')!=cfg['control']['protocol_sha256'] or payload.get('method')!=BUILD:
            raise ValueError('Resume checkpoint is not the matching deterministic control')
        load_checkpoint(model,optimizer,str(last),scheduler=scheduler)
        if payload.get('amp_scaler_state_dict'):scaler.load_state_dict(payload['amp_scaler_state_dict'])
        history=payload['history'];best_dice=float(payload['best_val_dice']);best_epoch=int(payload['best_epoch'])
        best_state=payload['best_model_state_dict'];patience=int(payload['patience'])
        next_epoch=int(payload['next_epoch']);progress=payload['partial_progress']
        complete=bool(payload['training_protocol_complete']);stop_reason=payload['training_stop_reason']
        if best_state is not None and not best.is_file():
            # This backup is for a lost best.pt, never a return to BALoRA weights.
            temp=_adapter_snapshot(model)
            model.load_state_dict(best_state,strict=False)
            save_checkpoint(model,None,best_epoch,str(best),adapter_only=True,extra={
                'config':cfg,'method':BUILD,'control_protocol_sha256':cfg['control']['protocol_sha256'],
                'best_val_dice':best_dice,'selection_metric':'fixed-prompt validation Dice only'})
            model.load_state_dict(temp,strict=False)
        del payload
    last_save=time.time()
    def save(epoch_index,partial,terminal=False):
        save_checkpoint(model,optimizer,epoch_index-1,str(last),adapter_only=True,scheduler=scheduler,
            extra={'config':cfg,'method':BUILD,'control_protocol_sha256':cfg['control']['protocol_sha256'],
                'next_epoch':epoch_index,'next_batch':int(partial.get('next_batch',0)),
                'partial_progress':dict(partial),'history':history,'best_val_dice':best_dice,
                'best_epoch':best_epoch,'best_model_state_dict':best_state,'patience':patience,
                'amp_scaler_state_dict':scaler.state_dict(),'training_protocol_complete':terminal,
                'training_stop_reason':stop_reason})
    if not last.is_file():save(0,{})
    epoch=next_epoch
    try:
        for epoch in range(next_epoch,target) if not complete else ():
            check_deadline(deadline,'control training before epoch')
            started=time.time()
            order=torch.randperm(len(train_loader.dataset),generator=torch.Generator().manual_seed(seed+epoch*1009)).tolist()
            next_batch=int(progress.get('next_batch',0))
            if not 0<=next_batch<=len(order):raise ValueError('Invalid partial-epoch cursor')
            remaining=order[next_batch:]
            loader=DataLoader(train_loader.dataset,batch_size=1,sampler=remaining,
                collate_fn=train_loader.collate_fn,num_workers=0,pin_memory=str(device).startswith('cuda'),
                generator=torch.Generator().manual_seed(seed+epoch))
            def callback(state):
                nonlocal last_save
                if time.time()-last_save>=float(tr.get('checkpoint_seconds',300)):
                    save(epoch,state);last_save=time.time()
            if remaining:
                loss=train_one_epoch(model,loader,optimizer,weights,1,device,epoch=epoch,
                    grad_clip_norm=tr['grad_clip_norm'],log_every=100,amp=tr['amp'],scaler=scaler,
                    deadline=deadline,progress_state=progress,progress_callback=callback,
                    start_step=next_batch,objective_fn=deterministic_objective)
            else:
                if int(progress.get('updates',0))!=len(order):raise ValueError('Training cursor/update count disagree')
                loss=progress['total_loss']/progress['updates']
            # A pause during validation must not repeat completed training steps.
            save(epoch,progress)
            val=evaluate_point_model(model,val_loader,device,allowed_cases=validation_cases,
                seed=seed+90001,progress_path=root/'validation_reports'/f'epoch_{epoch:02d}_progress.json',
                surface=False,deadline=deadline,split='validation')
            write_json(root/'validation_reports'/f'epoch_{epoch:02d}.json',val)
            if not val['complete']:
                stop_reason='BUDGET_PAUSED';save(epoch,progress);break
            score=val['segmentation']['dice']['mean']
            if score is None or not math.isfinite(score):raise FloatingPointError('No finite control validation Dice')
            if score>best_dice:
                best_dice=score;best_epoch=epoch;best_state=_adapter_snapshot(model);patience=0
                save_checkpoint(model,None,epoch,str(best),adapter_only=True,extra={
                    'config':cfg,'method':BUILD,'control_protocol_sha256':cfg['control']['protocol_sha256'],
                    'best_val_dice':best_dice,'selection_metric':'fixed-prompt validation Dice only',
                    'test_used_for_selection':False})
            else:patience+=1
            # Restored optimizers may have performed this epoch's updates in
            # the preceding kernel, before resuming a pending validation.
            if progress.get('updates',0)>0:optimizer._opt_called=True
            scheduler.step()
            history.append({'epoch':epoch,'train_loss':loss,'val_dice':score,
                'val_ece':val['calibration']['ece']['mean'],'optimizer_updates':progress['updates'],
                'amp_overflow_retries':progress.get('amp_overflow_retries',0),
                'seconds_this_session':time.time()-started,'lr_after_epoch':optimizer.param_groups[0]['lr']})
            next_epoch=epoch+1;progress={}
            complete=(next_epoch>=target or next_epoch>=tr['min_epochs'] and patience>=tr['early_stopping_patience'])
            stop_reason=('TARGET_EPOCHS_COMPLETED' if next_epoch>=target else 'EARLY_STOPPING_COMPLETED') if complete else 'TRAINING'
            save(next_epoch,{},complete)
            write_json(root/'training_history.json',history)
            print(f'control epoch={epoch} loss={loss:.5f} val_dice={score:.5f} best={best_dice:.5f} epochs_completed={next_epoch}',flush=True)
            if complete:break
    except BudgetExpired:
        stop_reason='BUDGET_PAUSED';save(epoch,progress)
    status={'build':BUILD,'protocol_complete':complete,'stop_reason':stop_reason,
        'requested_epochs':target,'completed_epochs':len(history),'best_epoch':best_epoch,
        'best_val_dice':best_dice if math.isfinite(best_dice) else None,
        'optimizer_updates':sum(r['optimizer_updates'] for r in history)+int(progress.get('updates',0)),
        'history':history,'best_checkpoint':str(best) if best.is_file() else None,
        'last_checkpoint':str(last),'test_evaluated_during_selection':False,
        'research_model':not model.test_backbone_injected}
    write_json(root/'training_status.json',status)
    return model,status


class IndexedView(Subset):
    def __init__(self,dataset,indices):
        super().__init__(dataset,indices)
        self._clip_index=[dataset._clip_index[i] for i in indices]
        self.case_ids=sorted({r[0] for r in self._clip_index})


def balanced_indices(dataset,per_case=1):
    by_case={cid:[] for cid in dataset.case_ids}
    for i,(cid,_,_) in enumerate(dataset._clip_index):by_case[cid].append(i)
    result=[]
    for cid,indices in by_case.items():
        if not indices:raise ValueError('A declared case has no foreground slices: '+cid)
        picks=[indices[min(len(indices)-1,int((j+.5)*len(indices)/per_case))] for j in range(per_case)]
        result.extend(dict.fromkeys(picks))
    return result


def profile_control(cfg,train_loader,val_loader,device,*,deadline,model=None):
    root=Path(cfg['experiment']['output_dir']);path=root/'profile_report.json'
    gpu=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else 'CPU_TEST'
    if path.is_file():
        old=json.loads(path.read_text())
        if old.get('complete') and old.get('protocol_sha256')==cfg['control']['protocol_sha256'] and old.get('gpu')==gpu and old.get('runtime_revision')==RUNTIME_REVISION and old.get('execution_mode','per_block')==cfg.get('runtime',{}).get('encoder_checkpointing','per_block'):
            return old
    set_seed(cfg['experiment']['seed'])
    if model is None:model=DeterministicSAM2(cfg,device)
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    before=_adapter_snapshot(model)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    scaler=_make_grad_scaler(bool(cfg['training']['amp'] and str(device).startswith('cuda')))
    samples=balanced_indices(train_loader.dataset)
    measured_train_cases=[train_loader.dataset._clip_index[i][0] for i in samples]
    elapsed=[]
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    for j,index in enumerate([samples[0],samples[-1],*samples]):
        check_deadline(deadline,'control profiling training')
        view=IndexedView(train_loader.dataset,[index])
        loader=DataLoader(view,batch_size=1,collate_fn=train_loader.collate_fn,
                          generator=torch.Generator().manual_seed(4300+j))
        sync();start=time.perf_counter()
        train_one_epoch(model,loader,optimizer,control_weights(cfg),1,device,amp=cfg['training']['amp'],
            scaler=scaler,deadline=deadline,log_every=10000,objective_fn=deterministic_objective)
        sync()
        if j>=2:elapsed.append(time.perf_counter()-start)
    changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(before[n],p.detach().cpu())]
    if not any('.image_encoder.' in n for n in changed) or not any('.sam_mask_decoder.' in n for n in changed):
        raise ValueError('Profiling did not update both encoder and decoder adapters')
    indices=balanced_indices(val_loader.dataset,per_case=6)
    view=IndexedView(val_loader.dataset,indices)
    sample_loader=DataLoader(view,batch_size=1,collate_fn=val_loader.collate_fn,
                             generator=torch.Generator().manual_seed(4301))
    profile_progress=root/'validation_reports'/'profiling_progress.json'
    # Profiling models are disposable. Repeating an interrupted diagnostic
    # starts a fresh profile, never mixes rows from another diagnostic model.
    profile_progress.unlink(missing_ok=True)
    val=evaluate_point_model(model,sample_loader,device,allowed_cases=view.case_ids,
        seed=cfg['experiment']['seed']+90001,progress_path=profile_progress,
        surface=False,deadline=deadline,split='profiling_validation')
    if not val['complete']:raise BudgetExpired('Profiling validation paused')
    p75=sorted(elapsed)[min(len(elapsed)-1,math.ceil(.75*len(elapsed))-1)]
    report={'complete':True,'protocol_sha256':cfg['control']['protocol_sha256'],'gpu':gpu,
        'runtime_revision':RUNTIME_REVISION,
        'execution_mode':cfg.get('runtime',{}).get('encoder_checkpointing','per_block'),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0,
        'training_sample_seconds':elapsed,'training_sample_cases':measured_train_cases,
        'training_p75_seconds_per_slice':p75,'train_foreground_slices':len(train_loader.dataset),
        'validation_sample_slices':len(indices),'validation_sample_seconds':val['elapsed_seconds_this_session'],
        'validation_foreground_slices':len(val_loader.dataset),
        'changed_trainable_tensors':len(changed),
        'profile_updates':len(samples)+2,'profile_model_discarded_before_research_training':True}
    write_json(path,report)
    # The caller also collects before constructing a new model. This scope
    # releases optimizer moments, graphs and the entire diagnostic backbone.
    del model,optimizer,scaler,before
    gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return report


def plan_control(profile,state,baseline_seconds,*,now=None,completed_epochs=0,training_complete=False):
    now=time.time() if now is None else float(now)
    train=profile['training_p75_seconds_per_slice']*profile['train_foreground_slices']
    val=profile['validation_sample_seconds']/max(1,profile['validation_sample_slices'])*profile['validation_foreground_slices']
    epoch=(train+val)*1.20
    if not math.isfinite(epoch) or epoch<=0 or not math.isfinite(baseline_seconds) or baseline_seconds<0:
        raise ValueError('Invalid control runtime measurements')
    final_reserve=max(1800.,float(baseline_seconds)*1.5)
    available=state['cumulative_work_deadline_unix']-now-final_reserve
    needed=0 if training_complete else max(0,20-int(completed_epochs))
    fits=available>=needed*epoch
    return {'complete':True,'estimated_epoch_seconds':epoch,'remaining_target_epochs':needed,
        'epoch_margin':.20,'budget_admission_revision':'control_budget_admission_v1_2',
        'requested_total_epochs':20,'completed_epochs_on_resume':int(completed_epochs),
        'training_protocol_already_complete':bool(training_complete),
        'estimated_final_evaluation_reserve_seconds':final_reserve,
        'estimated_training_seconds':needed*epoch,'available_training_seconds':max(0,available),
        'requested_protocol_estimated_to_fit':fits,
        'estimated_additional_sessions_needed':max(0,math.ceil(max(0,needed*epoch-(state['work_deadline_unix']-now))/(11*3600))),
        'training_deadline_unix':min(state['work_deadline_unix'],state['cumulative_work_deadline_unix']-final_reserve),
        'final_deadline_unix':state['work_deadline_unix'],
        'note':'Measured estimate with 20% epoch margin on p75 training time, at least 30 minutes for final evaluation, and 15-minute export reserve. The hard session/cumulative deadlines remain enforced; no data, resolution or epoch target is reduced.'}
