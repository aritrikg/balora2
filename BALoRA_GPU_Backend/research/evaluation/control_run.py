"""Budgeted deterministic control, then frozen three-method test comparison."""
from __future__ import annotations
import csv
import gc
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from datasets.btcv_dataset import BTCVDataset,collate_btcv
from evaluation.control_inputs import BUILD,protocol_digest
from evaluation.point_estimate import evaluate_point_model
from evaluation.validation_audit import sha256_file
from training.deterministic_control import run_fixed_training,profile_control,plan_control
from training.train import _read_checkpoint
from utils.checkpoint import load_checkpoint
from utils.control_session import export_control_bundle,finish_control_session,append_clean_ledger
from utils.runtime_budget import BudgetExpired,check_deadline,write_json


def make_loader(cfg,split):
    d=cfg['data'];key={'train':'train_split','validation':'val_split','test':'test_split'}[split]
    dataset=BTCVDataset(d['root_dir'],d[key],num_classes=d['num_classes'],clip_len=1,
        train=split=='train',image_size=d['image_size'],ct_window=tuple(d['ct_window']),
        cache_dir=d['cache_dir'],foreground_only=True)
    return DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_btcv,
                       generator=torch.Generator().manual_seed(cfg['experiment']['seed']),
                       pin_memory=torch.cuda.is_available())


def architecture_report(model,cfg):
    model.assert_control()
    trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    return {'method':BUILD,'encoder_rank':cfg['lora']['encoder_rank'],
        'decoder_rank':cfg['lora']['decoder_rank'],
        'encoder_adapter_layers':len(model.encoder_lora_layers),'decoder_adapter_layers':len(model.decoder_lora_layers),
        'trainable_tensors':len(trainable),'trainable_parameters':sum(trainable.values()),
        'all_trainable_parameters_are_deterministic_lora_factors':all(n.endswith(('.lora_A','.lora_B')) for n in trainable),
        'bayesian_adapters':0,'adaptive_rank_controller':False,
        'encoder_gradient_checkpointing':model.encoder_gradient_checkpointing_enabled,
        'encoder_checkpointing_mode':getattr(model,'encoder_gradient_checkpointing_mode',None),
        'test_backbone_injected':model.test_backbone_injected}


def export_three_way(context,control,training,cfg,root):
    if not training['protocol_complete'] or not control['complete']:
        raise ValueError('Partial training/evaluation cannot produce a completed comparison')
    actual={(o,c):r['n_slices'] for o,vols in control['per_volume_segmentation'].items() for c,r in vols.items()}
    if actual!=context['expected_counts']:
        raise ValueError('Control does not have identical volume-organ slice coverage')
    test=set(cfg['experiment']['split_case_ids']['test'])
    if set(control['protocol']['cases'])!=test or control['protocol']['seed']!=context['seed']:
        raise ValueError('Control test protocol changed')
    reports=[('SAM2 zero-shot',context['zero_shot']),
             (f"Deterministic LoRA, fixed rank {cfg['lora']['decoder_rank']}",control),
             ('BALoRA, no refinement',context['original_balora'])]
    rows=[]
    for label,r in reports:
        rows.append({'method':label,'dice':r['segmentation']['dice']['mean'],
            'dice_sd_volume_organ_pairs':r['segmentation']['dice']['std'],
            'n_volume_organ_pairs':r['segmentation']['dice']['n'],
            'iou':r['segmentation']['iou']['mean'],
            'hd95_pixels':r['segmentation'].get('hd95',{}).get('mean'),
            'assd_pixels':r['segmentation'].get('assd',{}).get('mean'),
            **{k:r['calibration'][k]['mean'] for k in ('ece','brier','nll')}})
    organs=[];volumes=[]
    for organ in sorted(control['per_organ_segmentation']):
        vals=[r['per_organ_segmentation'][organ]['dice']['mean'] for _,r in reports]
        organs.append({'organ':organ,'sam2_dice':vals[0],'deterministic_lora_dice':vals[1],
            'balora_dice':vals[2],'balora_minus_deterministic':vals[2]-vals[1]})
        for cid in sorted(control['per_volume_segmentation'][organ]):
            vals=[r['per_volume_segmentation'][organ][cid]['metrics']['dice'] for _,r in reports]
            volumes.append({'case_id':cid,'organ':organ,'sam2_dice':vals[0],
                'deterministic_lora_dice':vals[1],'balora_dice':vals[2],
                'balora_minus_deterministic':vals[2]-vals[1]})
    for name,data in [('method_comparison.csv',rows),('per_organ_comparison.csv',organs),('per_volume_comparison.csv',volumes)]:
        with (Path(root)/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    result={'build':BUILD,'complete':True,'methods':rows,
        'n_predictions_per_method':sum(actual.values()),'identical_volume_organ_slice_counts_verified':True,
        'test_cases':sorted(test),'prompt_seed':context['seed'],
        'control_protocol_sha256':cfg['control']['protocol_sha256'],
        'control_selected_checkpoint_sha256':sha256_file(Path(root)/'best.pt'),
        'original_balora_selected_checkpoint_sha256':context['reference']['checkpoint_sha256'],
        'validation_selected_refinement':'none','control_training':training,
        'scope':'Exploratory Fold-1 joint point-estimate/fixed-rank control after original test inspection. Does not isolate Bayesian sampling from adaptive rank individually. One training seed, not five-fold results. Test scores do not select checkpoints, rank, policies or epochs.'}
    write_json(Path(root)/'method_comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[r['organ'].replace('_',' ') for r in organs];x=list(range(len(organs)))
    fig,ax=plt.subplots(figsize=(16,6))
    for shift,key,label in [(-.25,'sam2_dice','SAM2 zero-shot'),(0,'deterministic_lora_dice','Deterministic LoRA (r=16)'),(.25,'balora_dice','BALoRA, no refinement')]:
        ax.bar([v+shift for v in x],[r[key] for r in organs],width=.25,label=label)
    ax.set_xticks(x,labels,rotation=55,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Dice')
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.17),ncol=3)
    fig.tight_layout();fig.savefig(Path(root)/'per_organ_dice.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return result


def run_control(context,cfg,state,*,device='cuda:0'):
    if not torch.cuda.is_available() or not str(device).startswith('cuda'):
        raise RuntimeError('Select a CUDA GPU for the real deterministic control')
    root=Path(cfg['experiment']['output_dir']);status='FAILED';result=None;archive=None;phase='setup'
    try:
        if state['protocol_sha256']!=protocol_digest(cfg):raise ValueError('Session/protocol identity mismatch')
        saved_comparison=root/'method_comparison.json'
        if saved_comparison.is_file():
            saved=json.loads(saved_comparison.read_text())
            if saved.get('complete') and saved.get('control_protocol_sha256')==cfg['control']['protocol_sha256']:
                if sha256_file(root/'best.pt')!=saved['control_selected_checkpoint_sha256']:
                    raise ValueError('Completed control checkpoint bytes changed')
                report=json.loads((root/'control_test_evaluation.json').read_text())
                counts={(o,c):r['n_slices'] for o,vols in report['per_volume_segmentation'].items() for c,r in vols.items()}
                if not report['complete'] or counts!=context['expected_counts']:
                    raise ValueError('Saved completed control test coverage changed')
                from evaluation.full_figures import export_full_figures
                if not (root/'figure_manifest.json').is_file():
                    phase='figures'
                    export_full_figures(root,context,deadline=state['work_deadline_unix'])
                status='DETERMINISTIC_CONTROL_COMPLETE';result=saved
                return result
        if sha256_file(cfg['sam2']['checkpoint'])!=cfg['sam2']['checkpoint_sha256']:
            raise ValueError('Official base checkpoint differs from the reference')
        check_deadline(state['work_deadline_unix'],'control setup')
        write_json(root/'control_config.json',cfg)
        train_loader=make_loader(cfg,'train');val_loader=make_loader(cfg,'validation')
        print('Control train cases:',train_loader.dataset.case_ids,'| foreground slices:',len(train_loader.dataset),flush=True)
        print('Control validation cases:',val_loader.dataset.case_ids,'| foreground slices:',len(val_loader.dataset),flush=True)
        if cfg['data']['num_classes']==14 and (len(train_loader.dataset)!=1347 or len(val_loader.dataset)!=371):
            raise ValueError('Original Fold-1 foreground slice counts changed')
        payload=_read_checkpoint(root/'last.pt') if (root/'last.pt').is_file() else {}
        if payload and (payload.get('control_protocol_sha256')!=cfg['control']['protocol_sha256'] or payload.get('method')!=BUILD):
            raise ValueError('Incompatible control checkpoint before profiling')
        execution=root/'runtime_execution.json'
        if execution.is_file():cfg['runtime']=json.loads(execution.read_text())
        training_complete=bool(payload.get('training_protocol_complete'))
        completed=len(payload.get('history',[]));del payload
        if training_complete:
            # Already completed research training is never profiled or repeated.
            profile={'training_p75_seconds_per_slice':1.,'train_foreground_slices':len(train_loader.dataset),
                'validation_sample_seconds':1.,'validation_sample_slices':1,
                'validation_foreground_slices':len(val_loader.dataset)}
        else:
            from training.control_execution import profile_execution
            profile=profile_execution(cfg,train_loader,val_loader,device,deadline=state['work_deadline_unix'],profile_fn=profile_control)
            write_json(execution,cfg.get('runtime',{}))
            write_json(root/'control_config.json',cfg)
        gc.collect();torch.cuda.empty_cache()
        baseline_seconds=float(context['zero_shot']['elapsed_seconds_this_session'])
        plan=plan_control(profile,state,baseline_seconds,completed_epochs=completed,training_complete=training_complete)
        write_json(root/'runtime_plan.json',plan)
        print('Measured runtime plan:',json.dumps(plan,indent=2),flush=True)
        if not plan['requested_protocol_estimated_to_fit']:
            status='BUDGET_PLAN_REJECTED'
            result={'complete':False,'status':status,'runtime_plan':plan,
                'reason':'Measured full target and final reserve do not fit the remaining cumulative budget; no epoch target or data coverage was reduced.'}
            return result
        model,training=run_fixed_training(cfg,train_loader,val_loader,device,deadline=plan['training_deadline_unix'])
        write_json(root/'architecture_report.json',architecture_report(model,cfg))
        if not training['protocol_complete']:
            status='PARTIAL_CONTROL_TRAINING_RESUME_REQUIRED'
            result={'complete':False,'status':status,'training':training}
            return result
        load_checkpoint(model,None,str(root/'best.pt'),restore_rng=False)
        write_json(root/'selected_control_checkpoint.json',{
            'sha256':sha256_file(root/'best.pt'),'epoch':training['best_epoch'],
            'best_validation_dice':training['best_val_dice'],'selection':'fixed-prompt validation Dice only',
            'test_used_for_selection':False,'control_protocol_sha256':cfg['control']['protocol_sha256']})
        # This is the first point at which any held-out BTCV volume is opened.
        test_loader=make_loader(cfg,'test')
        evaluation=evaluate_point_model(model,test_loader,device,
            allowed_cases=context['reference']['splits']['test'],seed=context['seed'],
            progress_path=root/'control_test_progress.json',surface=True,
            deadline=plan['final_deadline_unix'],split='test')
        evaluation['gpu']=torch.cuda.get_device_name(0);evaluation['torch']=torch.__version__
        write_json(root/'control_test_evaluation.json',evaluation)
        if not evaluation['complete']:
            status='PARTIAL_CONTROL_TEST_RESUME_REQUIRED'
            result={'complete':False,'status':status,'training':training}
            return result
        result=export_three_way(context,evaluation,training,cfg,root)
        del model
        gc.collect();torch.cuda.empty_cache()
        from evaluation.full_figures import export_full_figures
        phase='figures'
        export_full_figures(root,context,deadline=plan['final_deadline_unix'])
        status='DETERMINISTIC_CONTROL_COMPLETE'
        print(json.dumps(result['methods'],indent=2),flush=True)
        return result
    except BudgetExpired as exc:
        status='PARTIAL_CONTROL_FIGURES_RESUME_REQUIRED' if phase=='figures' else 'PARTIAL_CONTROL_PROFILING_RESUME_REQUIRED'
        result={'complete':False,'status':status,'reason':str(exc)}
        return result
    except Exception as exc:
        write_json(root/'failure.json',{'build':BUILD,'exception':type(exc).__name__,'message':str(exc),
            'last_consistent_checkpoint':str(root/'last.pt') if (root/'last.pt').is_file() else None})
        raise
    finally:
        write_json(root/'control_status.json',{'build':BUILD,'execution_status':status,
            'complete':status=='DETERMINISTIC_CONTROL_COMPLETE',
            'control_protocol_sha256':cfg['control']['protocol_sha256']})
        try:
            archive=export_control_bundle(root)
        finally:
            closed=finish_control_session(root,state,status)
            if archive is not None:append_clean_ledger(archive,root)
            print('Control status:',status,'| cumulative accounted hours:',
                round(closed['accounted_runtime_seconds']/3600,4),flush=True)
            if archive is not None:print('Portable control ZIP:',archive,flush=True)
