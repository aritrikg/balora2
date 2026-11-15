"""Measure a memory/runtime policy before constructing fresh research weights."""
import gc
from pathlib import Path
import torch
from training.deterministic_control import profile_control


def profile_execution(cfg,train_loader,val_loader,device,*,deadline,profile_fn=None):
    cfg.setdefault('runtime',{'encoder_checkpointing':'per_block'})
    mode=cfg['runtime']['encoder_checkpointing']
    if mode not in ('off','per_block'):raise ValueError('Invalid encoder checkpointing mode')
    def measure():
        return (profile_fn or profile_control)(cfg,train_loader,val_loader,device,deadline=deadline)
    fallback=False
    try:
        result=measure()
        if mode=='off' and str(device).startswith('cuda'):
            total=torch.cuda.get_device_properties(torch.device(device)).total_memory
            fallback=result['peak_reserved_bytes']>.85*total
    except torch.cuda.OutOfMemoryError:
        if mode!='off':raise
        # Do not retain the failed traceback and its model/graphs while retrying.
        fallback=True
    if fallback:
        gc.collect();torch.cuda.empty_cache()
        cfg['runtime']['encoder_checkpointing']='per_block'
        (Path(cfg['experiment']['output_dir'])/'profile_report.json').unlink(missing_ok=True)
        print('Activation storage exceeded the memory allowance; profiling per-block checkpointing.',flush=True)
        result=measure()
    return result
