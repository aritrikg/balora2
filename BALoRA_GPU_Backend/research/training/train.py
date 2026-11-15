"""BALoRA training with per-organ supervision, validation, and resume-ready logs."""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
from typing import Any
import torch
import yaml
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # explicit no-op fallback for environments without tensorboard
        def __init__(self, *a, **kw):
            self.log_dir = str(a[0]) if a else None
        def add_scalar(self, *a, **kw):
            return None
        def flush(self):
            return None
        def close(self):
            return None

from datasets.btcv_dataset import BTCVDataset, collate_btcv
from datasets.splits import make_fold_split, write_fold_splits
from losses.losses import LossWeights, compute_total_loss
from models.sam2_wrapper import Sam2BayesianLoRAConfig, Sam2BayesianLoRAModel
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.prompting import sample_foreground_points
from utils.runtime_budget import BudgetExpired, check_deadline, write_json
from utils.session_runtime import checkpoint_runtime_fields, training_is_complete


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = [("data", "image_size"), ("data", "num_classes"),
                ("bayesian", "k_train"), ("bayesian", "k_infer"),
                ("training", "batch_size"), ("training", "num_epochs")]
    missing = [f"{a}.{b}" for a,b in required if a not in cfg or b not in cfg[a]]
    if missing:
        raise KeyError(f"Missing required configuration fields: {missing}")
    if int(cfg["data"]["image_size"]) <= 0 or int(cfg["data"]["num_classes"]) < 2:
        raise ValueError("data.image_size must be positive and num_classes >=2")
    if int(cfg["bayesian"]["k_train"]) < 2 or int(cfg["bayesian"]["k_infer"]) < 2:
        raise ValueError("bayesian.k_train and k_infer must be >=2 because epistemic uncertainty needs multiple samples")
    if int(cfg["training"]["batch_size"]) < 1 or int(cfg["training"]["num_epochs"]) < 1:
        raise ValueError("training.batch_size and num_epochs must be positive")


def build_model(cfg: dict[str, Any], device: str) -> Sam2BayesianLoRAModel:
    s, l, b = cfg["sam2"], cfg["lora"], cfg["bayesian"]
    mc = Sam2BayesianLoRAConfig(
        model_cfg=s["model_cfg"], checkpoint=s["checkpoint"],
        target_modules=l["target_modules"], rank_bank=l["rank_bank"],
        alpha=l.get("alpha", 16), dropout=l.get("dropout", 0.0),
        prior_std=b.get("prior_std", 1.0), init_posterior_std=b.get("init_posterior_std", 0.05),
        device=device, encoder_target_modules=l.get("encoder_target_modules"),
        point_estimate_module_substrings=b.get("point_estimate_module_substrings", ["image_encoder"]),
        attribution_gamma=l.get("attribution_gamma", 4.0),
        learned_prior_weight=l.get("learned_prior_weight", 0.2))
    model = Sam2BayesianLoRAModel(mc).to(device)
    model.encoder_amp = bool(cfg['training'].get('amp', False))
    model._run_config = cfg
    return model


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    d, t = cfg["data"], cfg["training"]
    train_ds = BTCVDataset(d["root_dir"], d["train_split"], d["num_classes"],
                           clip_len=d.get("clip_len", 1), train=True, image_size=d["image_size"],
                           ct_window=tuple(d.get("ct_window", [-175, 250])),
                           cache_dir=d.get('cache_dir'), foreground_only=d.get('foreground_only', True))
    val_ds = BTCVDataset(d["root_dir"], d["val_split"], d["num_classes"],
                         clip_len=d.get("clip_len", 1), train=False, image_size=d["image_size"],
                         ct_window=tuple(d.get("ct_window", [-175, 250])),
                         cache_dir=d.get('cache_dir'), foreground_only=d.get('foreground_only', True))
    common = dict(batch_size=int(t["batch_size"]), collate_fn=collate_btcv,
                  num_workers=int(t.get("num_workers", 0)), pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, shuffle=False, **common), DataLoader(val_ds, shuffle=False, **common))


def _unpack_valid_slices(batch: dict[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    x = batch["image"].to(device, non_blocking=True)      # B,T,1,H,W
    y = batch["label"].to(device, non_blocking=True)      # B,T,C,H,W
    b,t = x.shape[:2]
    valid = batch.get("valid_slice_mask", torch.ones((b,t), dtype=torch.bool)).reshape(-1).to(device)
    x = x.reshape(b*t, *x.shape[2:])[valid]
    y = y.reshape(b*t, *y.shape[2:])[valid]
    ids: list[str] = []; z_indices: list[int] = []
    for i, cid in enumerate(batch["case_ids"]):
        for z in batch["slice_indices"][i]:
            if z >= 0:
                ids.append(cid); z_indices.append(int(z))
    return x, y, ids, z_indices


def _make_grad_scaler(enabled: bool) -> Any:
    """Create a persistent-capable FP16 GradScaler across supported PyTorch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled), init_scale=1024.0,
                                    growth_factor=2.0, backoff_factor=0.5,
                                    growth_interval=2000)
    except (AttributeError, TypeError):  # compatibility fallback for older PyTorch
        return torch.cuda.amp.GradScaler(enabled=bool(enabled), init_scale=1024.0,
                                         growth_factor=2.0, backoff_factor=0.5,
                                         growth_interval=2000)


def _promote_embedding_features_to_fp32(embedding: dict[str, Any]) -> dict[str, Any]:
    """Keep SAM2 feature geometry/gradients, but hand full-precision tensors to the decoder."""
    if torch.is_tensor(embedding.get("image_embed")):
        embedding["image_embed"] = embedding["image_embed"].float()
    if "high_res_feats" in embedding:
        embedding["high_res_feats"] = [x.float() if torch.is_tensor(x) else x
                                        for x in embedding["high_res_feats"]]
    return embedding


def _clear_adapter_graphs(model):
    for module in model.modules():
        if hasattr(module, 'last_adapter_output'): module.last_adapter_output = None


def _non_finite_gradient_names(model):
    """One host synchronization per device on the usual finite-gradient path."""
    groups={}
    for name,p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            groups.setdefault(p.grad.device,[]).append((name,p.grad))
    bad=[]
    for pairs in groups.values():
        finite=torch.stack([torch.isfinite(g).all() for _,g in pairs])
        if not bool(finite.all()):
            flags=finite.detach().cpu().tolist()
            bad.extend(name for (name,_),flag in zip(pairs,flags) if not flag)
    return bad


def _batch_objective(model, images, labels, weights, k_train, encoder_amp, kl_count):
    if encoder_amp:
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            embedding=model.encode_image(images)
        embedding=_promote_embedding_features_to_fp32(embedding)
    else:
        embedding=model.encode_image(images)
    class_seg=[];class_cal=[];zero=images.new_zeros(())
    for class_id in range(1,labels.shape[1]):
        present=labels[:,class_id].flatten(1).any(1)
        if not bool(present.any()): continue
        target=labels[present,class_id].unsqueeze(1)
        points=sample_foreground_points(target[:,0])
        point_labels=torch.ones(points.shape[:2],dtype=torch.int32,device=images.device)
        logits,_probe,_ranks=model.decode_with_adaptive_rank(embedding[present],points,point_labels,k_samples=k_train)
        mean_prob=torch.sigmoid(logits.float()).mean(0)
        mean_logits=torch.logit(mean_prob.clamp(1e-6,1-1e-6))
        loss=compute_total_loss(mean_logits,target.float(),zero,zero,weights)
        if not torch.isfinite(loss.seg) or not torch.isfinite(loss.calibration):
            raise FloatingPointError(f'Non-finite loss for organ class {class_id}')
        class_seg.append(loss.seg);class_cal.append(loss.calibration)
    if not class_seg: raise ValueError('No organ foreground reached training objective')
    seg=torch.stack(class_seg).mean();cal=torch.stack(class_cal).mean()
    kl=model.total_kl()/max(1,kl_count);rank=model.rank_controller.sparsity_loss()
    total=seg+weights.lambda_kl*kl+weights.lambda_cal*cal+weights.lambda_rank*rank
    return total,{'seg':seg,'cal':cal,'kl':kl,'rank':rank}


def train_one_epoch(model: Sam2BayesianLoRAModel, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, weights: LossWeights,
                    k_train: int, device: str, writer: Any = None, epoch: int = 0,
                    grad_clip_norm: float = 1.0, log_every: int = 10,
                    amp: bool = False, scaler: Any = None, deadline: float | None = None,
                    progress_state: dict | None = None, progress_callback: Any = None,
                    start_step: int = 0, objective_fn: Any = None) -> float:
    """FP16 encoder/FP32 decoder with exact-batch AMP overflow retries.

    GradScaler growth can eventually overflow scaled FP16 gradients despite
    finite unscaled objectives. Halve the scale, restore the batch's RNG and
    recompute the SAME batch. Never count a skipped optimizer step as an
    update. Genuine non-finite loss/FP32 gradients still fail explicitly.
    """
    if k_train<2 and objective_fn is None: raise ValueError('k_train >=2 is required')
    if k_train<1: raise ValueError('At least one prediction is required')
    objective = _batch_objective if objective_fn is None else objective_fn
    model.train()
    use_amp=bool(amp and torch.cuda.is_available() and torch.device(device).type=='cuda')
    if scaler is None: scaler=_make_grad_scaler(use_amp)
    elif use_amp and not scaler.is_enabled(): raise ValueError('Enabled GradScaler required for CUDA FP16')
    print('Training precision:','FP16 encoder + GradScaler; FP32 decoder/loss' if use_amp else 'FP32')
    progress=progress_state if progress_state is not None else {}
    total_running=float(progress.get('total_loss',0));updates=int(progress.get('updates',0))
    skipped=int(progress.get('skipped',0));overflow_retries=int(progress.get('amp_overflow_retries',0))
    kl_count=getattr(loader.dataset,'total_clips_before_filter',len(loader.dataset)) if hasattr(loader,'dataset') else 1
    trainable=[p for p in model.parameters() if p.requires_grad]
    started=time.time()
    for step,batch in enumerate(loader):
        check_deadline(deadline,'training')
        images,labels,_,_=_unpack_valid_slices(batch,device)
        if images.shape[0]==0 or not bool(labels[:,1:].any()):
            skipped+=1
            progress.update(skipped=skipped,next_batch=start_step+step+1)
            continue
        # Only torch RNG is used by point sampling, posterior draws and LoRA
        # dropout. Preserve it so overflow retries do not change the objective.
        rng_cpu=torch.get_rng_state() if use_amp else None
        rng_cuda=torch.cuda.get_rng_state(device) if use_amp else None
        for attempt in range(17):
            check_deadline(deadline,'training batch retry')
            optimizer.zero_grad(set_to_none=True);_clear_adapter_graphs(model)
            loss,terms=objective(model,images,labels,weights,k_train,use_amp,kl_count)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite objective epoch={epoch} batch={start_step+step}')
            scaler.scale(loss).backward()
            if use_amp: scaler.unscale_(optimizer)
            bad=_non_finite_gradient_names(model)
            if bad:
                optimizer.zero_grad(set_to_none=True);_clear_adapter_graphs(model)
                del loss,terms
                if not use_amp or attempt==16:
                    raise FloatingPointError('Persistent non-finite gradients: '+', '.join(bad[:8]))
                old_scale=float(scaler.get_scale())
                scaler.update(new_scale=old_scale*0.5)  # resets per-optimizer unscale state
                overflow_retries+=1
                torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda,device)
                print(f'AMP overflow retry {attempt+1}: scale {old_scale:g} -> {scaler.get_scale():g}; same batch, no optimizer update',flush=True)
                continue
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable,float(grad_clip_norm)) if grad_clip_norm>0 else torch.tensor(0.)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError('Non-finite gradient norm')
            scaler.step(optimizer);scaler.update()
            break
        updates+=1;total_running+=float(loss.detach())
        progress.update(total_loss=total_running,updates=updates,skipped=skipped,
            amp_overflow_retries=overflow_retries,next_batch=start_step+step+1)
        if writer is not None and step%max(1,log_every)==0:
            gs=epoch*max(1,len(loader))+step
            writer.add_scalar('train/loss_total',float(loss.detach()),gs)
            for key,value in terms.items(): writer.add_scalar('train/loss_'+key,float(value.detach()),gs)
            writer.add_scalar('train/grad_norm',float(grad_norm),gs)
            writer.add_scalar('train/amp_overflow_retries',overflow_retries,gs)
        if (step+1)%max(1,log_every)==0:
            print(f'epoch={epoch} batch={start_step+step+1} updates={updates} loss={total_running/updates:.5f} elapsed_min={(time.time()-started)/60:.1f}',flush=True)
        # Release old graphs before checkpointing or the next encoder call.
        _clear_adapter_graphs(model)
        del loss,terms,images,labels,grad_norm
        if progress_callback is not None: progress_callback(progress)
    if updates==0:
        raise RuntimeError(f'Epoch {epoch} had zero optimizer updates; skipped_batches={skipped}')
    model.last_training_stats={'updates':updates,'skipped':skipped,'amp_overflow_retries':overflow_retries}
    epoch_loss=total_running/updates
    print(f'epoch {epoch}: mean_loss={epoch_loss:.6f}; optimizer_updates={updates}; skipped_batches={skipped}; AMP_retries={overflow_retries}; seconds={time.time()-started:.1f}',flush=True)
    return epoch_loss


def _read_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _adapter_snapshot(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def _assert_resume_identity(saved, cfg):
    if not isinstance(saved, dict):
        raise ValueError('Checkpoint has no experiment config; cannot verify a research resume')
    # Paths may change when outputs are attached to a fresh Kaggle session.
    paths = [('data','image_size'), ('data','num_classes'), ('data','ct_window'),
             ('lora','target_modules'), ('lora','encoder_target_modules'), ('lora','rank_bank'),
             ('lora','alpha'), ('lora','dropout'), ('bayesian','k_train'), ('bayesian','k_infer'),
             ('bayesian','prior_std'), ('bayesian','init_posterior_std'), ('sam2','model_cfg'),
             ('experiment','split_hash'), ('experiment','seed'), ('training','batch_size'),
             ('lora','attribution_gamma'), ('lora','learned_prior_weight')]
    for section,key in paths:
        if saved.get(section,{}).get(key) != cfg.get(section,{}).get(key):
            raise ValueError(f'Incompatible resume setting: {section}.{key}')
    for key in ('checkpoint_sha256','resolved_commit'):
        old = saved.get('sam2',{}).get(key)
        if old and old != cfg.get('sam2',{}).get(key):
            raise ValueError(f'Frozen SAM2 mismatch: {key}')


def run_training(cfg: dict[str, Any], resume_from: str | None = None) -> dict[str, Any]:
    validate_config(cfg)
    seed = int(cfg['experiment'].get('seed',42)); set_seed(seed)
    device = cfg['experiment'].get('device','cuda') if torch.cuda.is_available() else 'cpu'
    output = Path(cfg['experiment'].get('output_dir','experiments/balora'))
    output.mkdir(parents=True, exist_ok=True)
    if not resume_from and any((output/n).exists() for n in ('best.pt','last.pt','training_history.json')):
        raise FileExistsError('Existing training output: resume from last.pt or use a new BALORA_RUN_ID')
    write_json(output/'resolved_config.json', cfg)
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=cfg['training']['lr'], weight_decay=cfg['training'].get('weight_decay',1e-5))
    target_epochs = int(cfg['training']['num_epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=target_epochs)
    scaler = _make_grad_scaler(bool(cfg['training'].get('amp',False) and str(device).startswith('cuda')))
    start_epoch = 0; next_batch = 0; progress = {}
    history = {'epoch':[], 'train_loss':[], 'val_dice':[], 'val_ece':[], 'epoch_seconds':[]}
    best_dice = -float('inf'); patience = 0; best_state = None; best_epoch = -1
    schedule_note = None
    selection_policy = 'fixed_by_case_slice_organ_v1'
    resumed_protocol_complete = False
    best_path, last_path = output/'best.pt', output/'last.pt'
    if resume_from:
        payload = _read_checkpoint(resume_from)
        resumed_protocol_complete = training_is_complete(payload)
        _assert_resume_identity(payload.get('config'), cfg)
        start_epoch = load_checkpoint(model, optimizer, resume_from, map_location='cpu', scheduler=scheduler)
        # optimizer.load_state_dict moves states onto their parameter devices.
        if payload.get('amp_scaler_state_dict'): scaler.load_state_dict(payload['amp_scaler_state_dict'])
        history = payload.get('history',history)
        history.setdefault('epoch_seconds', [])
        best_dice = float(payload.get('best_val_dice',-float('inf')))
        patience = int(payload.get('patience',0))
        next_batch = int(payload.get('next_batch',0))
        progress = dict(payload.get('partial_progress',{})) if next_batch else {}
        best_epoch = int(payload.get('best_epoch',-1))
        best_state = payload.get('best_model_state_dict')
        selection_policy = payload.get('selection_seed_policy')
        if best_state is None:
            sibling = Path(resume_from).parent/'best.pt'
            if Path(resume_from).name == 'best.pt' or (history.get('epoch') and
                    history.get('val_dice',[])[-1] == best_dice and not next_batch):
                best_state = _adapter_snapshot(model); best_epoch = start_epoch-1
            elif sibling.is_file():
                bp = _read_checkpoint(sibling)
                names = {n for n,p in model.named_parameters() if p.requires_grad}
                best_state = {k:v for k,v in bp['model_state_dict'].items() if k in names}
                best_epoch = int(bp['epoch'])
            elif math.isfinite(best_dice):
                raise ValueError('Resume needs the selected best.pt alongside this legacy checkpoint')
        if best_state is not None:
            torch.save({'epoch':best_epoch,'model_state_dict':best_state,'checkpoint_format':'adapters_v1',
                        'config':cfg,'best_val_dice':best_dice}, best_path.with_suffix('.tmp'))
            os.replace(best_path.with_suffix('.tmp'),best_path)
        old_target = int(payload.get('config',{}).get('training',{}).get('num_epochs',target_epochs))
        if old_target != target_epochs:
            resumed_protocol_complete = False
            # Extending the original six-epoch cosine schedule otherwise resumes
            # at near-zero LR. This explicit continuation phase is recorded.
            continuation_lr = float(cfg['training'].get('continuation_lr',cfg['training']['lr']*0.25))
            for group in optimizer.param_groups:
                group['lr'] = continuation_lr; group['initial_lr'] = continuation_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,target_epochs-start_epoch))
            schedule_note = f'Continuation schedule: {old_target} -> {target_epochs} total epochs; LR={continuation_lr}'
            print(schedule_note)
        del payload
    writer = SummaryWriter(str(output/'tensorboard'))
    history.setdefault('amp_overflow_retries',[None]*len(history.get('epoch',[])))
    weights = LossWeights(**{k:cfg['loss'][k] for k in
        ('dice_weight','bce_weight','lambda_kl','lambda_cal','lambda_rank') if k in cfg['loss']})
    deadline = cfg.get('runtime',{}).get('training_deadline_unix')
    pause_reason = cfg.get('runtime',{}).get('training_pause_reason','TIME_BUDGET_PAUSED')
    min_epochs = int(cfg['training'].get('min_epochs',1))
    stop_reason = 'TARGET_EPOCHS_COMPLETED'
    started = time.time(); recent_epoch = None; last_save_time = time.time()

    def save_resume(epoch, state=None, complete=False):
        state = state or {}
        save_checkpoint(model,optimizer,epoch,str(last_path),scheduler=scheduler,adapter_only=True,
            extra={'history':history,'best_val_dice':best_dice,'best_epoch':best_epoch,
                   'best_model_state_dict':best_state,'patience':patience,'config':cfg,
                   'amp_scaler_state_dict':scaler.state_dict(), 'next_batch':int(state.get('next_batch',0)),
                   'partial_progress':dict(state), 'schedule_note':schedule_note,
                   'selection_seed_policy':selection_policy,
                   'training_protocol_complete':complete,
                   'training_stop_reason':stop_reason,
                   **checkpoint_runtime_fields(cfg.get('runtime',{}))})

    epoch = start_epoch
    try:
        if resumed_protocol_complete:
            stop_reason = 'EARLY_STOPPING_COMPLETED' if start_epoch < target_epochs else 'TARGET_EPOCHS_COMPLETED'
        if best_state is not None and selection_policy != 'fixed_by_case_slice_organ_v1':
            # Original notebook changed validation prompts between epochs.
            # Re-score its selected model using the NEW fixed validation seed
            # before comparing new checkpoints. Never involve test volumes.
            from evaluation.evaluate import run_evaluation
            current_state = _adapter_snapshot(model)
            try:
                model.load_state_dict(best_state, strict=False)
                baseline = run_evaluation(model,val_loader,device,k_infer=cfg['bayesian']['k_infer'],
                    compute_surface_distances=False,seed=seed+90001,deadline=deadline)
                if not baseline['protocol']['complete']:
                    raise BudgetExpired('Legacy checkpoint revalidation paused')
                best_dice = baseline['segmentation']['dice']['mean']
                if best_dice is None: raise ValueError('Legacy checkpoint validation produced no Dice')
                selection_policy = 'fixed_by_case_slice_organ_v1'
                write_json(output/'legacy_checkpoint_revalidation.json',baseline)
                torch.save({'epoch':best_epoch,'model_state_dict':best_state,
                    'checkpoint_format':'adapters_v1','config':cfg,'best_val_dice':best_dice,
                    'selection_seed_policy':selection_policy},best_path.with_suffix('.tmp'))
                os.replace(best_path.with_suffix('.tmp'),best_path)
                print(f'Legacy selected model revalidated with fixed prompts: Dice={best_dice:.5f}')
            finally:
                model.load_state_dict(current_state, strict=False)
                del current_state
        for epoch in range(start_epoch,target_epochs) if not resumed_protocol_complete else ():
            check_deadline(deadline,'training before epoch')
            # Refuse another epoch when recent COMPLETE train+validation timing
            # no longer fits. This protects final test time without relabeling
            # the requested target as a completed smaller experiment.
            if pause_reason != 'SESSION_LIMIT_PAUSED' and recent_epoch and deadline and time.time()+recent_epoch*1.20 > deadline:
                stop_reason = pause_reason; break
            epoch_started = time.time()
            generator = torch.Generator().manual_seed(seed+epoch*1009)
            order = torch.randperm(len(train_loader.dataset),generator=generator).tolist()
            bs = int(cfg['training']['batch_size'])
            remaining = order[next_batch*bs:]
            epoch_loader = DataLoader(train_loader.dataset,batch_size=bs,sampler=remaining,
                collate_fn=train_loader.collate_fn,num_workers=0,pin_memory=str(device).startswith('cuda'),
                generator=torch.Generator().manual_seed(seed+epoch))
            def callback(state):
                nonlocal last_save_time
                if time.time()-last_save_time >= float(cfg['training'].get('checkpoint_seconds',900)):
                    save_resume(epoch-1,state); last_save_time=time.time()
            if remaining:
                train_loss = train_one_epoch(model,epoch_loader,optimizer,weights,
                    cfg['bayesian']['k_train'],device,writer,epoch,
                    cfg['training'].get('grad_clip_norm',1.0),cfg.get('logging',{}).get('log_every_n_steps',100),
                    amp=cfg['training'].get('amp',False),scaler=scaler,deadline=deadline,
                    progress_state=progress,progress_callback=callback,start_step=next_batch)
            else:
                train_loss = progress['total_loss']/max(1,progress['updates'])
            from evaluation.evaluate import run_evaluation
            val_result = run_evaluation(model,val_loader,device,k_infer=cfg['bayesian']['k_infer'],
                refinement_config=None,compute_surface_distances=False,
                seed=seed+90001,deadline=deadline)
            if not val_result['protocol']['complete']:
                save_resume(epoch-1,progress); stop_reason=pause_reason; break
            val_dice = val_result['segmentation'].get('dice',{}).get('mean')
            val_ece = val_result['calibration'].get('ece',{}).get('mean')
            if val_dice is None or not math.isfinite(val_dice):
                raise FloatingPointError('No finite validation Dice; checkpoint selection is invalid')
            scheduler.step()
            for key,value in [('epoch',epoch),('train_loss',train_loss),('val_dice',val_dice),('val_ece',val_ece)]:
                history[key].append(value)
            recent_epoch = time.time()-epoch_started
            history['epoch_seconds'].append(recent_epoch)
            history['amp_overflow_retries'].append(int(progress.get('amp_overflow_retries',0)))
            if val_dice > best_dice:
                best_dice=val_dice; patience=0; best_epoch=epoch; best_state=_adapter_snapshot(model)
                save_checkpoint(model,optimizer,epoch,str(best_path),scheduler=scheduler,adapter_only=True,
                    extra={'best_val_dice':best_dice,'best_epoch':best_epoch,'history':history,'patience':0,
                           'config':cfg,'amp_scaler_state_dict':scaler.state_dict(),
                           'selection_seed_policy':selection_policy})
            else: patience += 1
            next_batch=0; progress={}
            save_resume(epoch)
            write_json(output/'training_history.json',history)
            writer.add_scalar('validation/dice',val_dice,epoch)
            print(f'epoch={epoch} loss={train_loss:.5f} val_dice={val_dice:.5f} best={best_dice:.5f} epoch_min={recent_epoch/60:.1f}',flush=True)
            if epoch+1 >= min_epochs and patience >= int(cfg['training'].get('early_stopping_patience',6)):
                stop_reason='EARLY_STOPPING_COMPLETED'; break
    except BudgetExpired:
        save_resume(epoch-1,progress); stop_reason=pause_reason
    except Exception as exc:
        # Retain the most recent consistent periodic checkpoint. The current
        # batch may have failed halfway through and must not be saved as valid.
        write_json(output/'failure.json',{'stage':'training','exception':type(exc).__name__,
            'message':str(exc),'last_consistent_checkpoint':str(last_path) if last_path.exists() else None})
        raise
    finally:
        writer.flush(); writer.close()
    # A budget stop before any update still writes an initial resumable state.
    if not last_path.is_file(): save_resume(start_epoch-1,progress)
    completed = stop_reason in ('TARGET_EPOCHS_COMPLETED','EARLY_STOPPING_COMPLETED')
    if start_epoch >= target_epochs: completed=True
    if completed:
        # Persist terminal early stopping/target status before final evaluation,
        # so another session does not retrain an already selected model.
        final_epoch=int(history['epoch'][-1]) if history['epoch'] else start_epoch-1
        save_resume(final_epoch,complete=True)
    result = {'model':model,'history':history,'best_checkpoint':str(best_path) if best_path.exists() else None,
              'last_checkpoint':str(last_path),'best_epoch':best_epoch,'best_val_dice':best_dice,
              'stop_reason':stop_reason,'protocol_complete':completed,'requested_epochs':target_epochs,
              'completed_epochs':len(history['epoch']),'elapsed_seconds':time.time()-started,
              **checkpoint_runtime_fields(cfg.get('runtime',{})),
              'continuation_schedule':schedule_note}
    write_json(output/'training_status.json',{k:v for k,v in result.items() if k != 'model'})
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--resume",default=None)
    args=parser.parse_args(); run_training(load_config(args.config), resume_from=args.resume)

if __name__ == "__main__": main()
