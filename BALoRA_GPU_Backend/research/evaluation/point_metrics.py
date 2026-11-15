"""Single-pass CPU metric bookkeeping for native point-estimate masks.

The binary formulas, smoothing, ECE bin boundaries and NLL clamp match the
original metric definitions. Histogram reductions may differ in the final
FP32 rounding of calibration scores. Dice, used for selection, retains FP32
scalar arithmetic. HD95 and ASSD share the same two distance transforms.
"""
from __future__ import annotations
import numpy as np
import torch
from evaluation.metrics import SegmentationMetrics,CalibrationMetrics

REVISION='point_metrics_single_pass_v1_1'


def _surface_pair(pred,target):
    if not pred.any() and not target.any():return 0.,0.
    if not pred.any() or not target.any():return float('inf'),float('inf')
    from scipy.ndimage import binary_erosion,distance_transform_edt
    ps=pred & ~binary_erosion(pred);ts=target & ~binary_erosion(target)
    if not ps.any() or not ts.any():return float('inf'),float('inf')
    pd=distance_transform_edt(~ps);td=distance_transform_edt(~ts)
    distances=np.concatenate((td[ps],pd[ts]))
    return float(np.percentile(distances,95)),float(distances.mean())


def _hard_ece(p,y,n_bins):
    if n_bins<1:raise ValueError('n_bins must be positive')
    if not p.size:return 0.
    edges=np.linspace(0.,1.,n_bins+1)
    # Original bins: [0,e1], (e1,e2], ..., (eN-1,1].
    index=np.searchsorted(edges,p,side='left')-1
    index=np.where(p==0,0,index)
    valid=(p>=0)&(p<=1)
    index=index[valid]
    count=np.bincount(index,minlength=n_bins)
    p_sum=np.bincount(index,weights=p[valid],minlength=n_bins)
    y_sum=np.bincount(index,weights=y[valid],minlength=n_bins)
    used=count>0
    # Mean(confidence)-mean(label), weighted by bin mass, equals
    # abs(sum(confidence)-sum(label))/N within each populated bin.
    return float(np.abs(p_sum[used]-y_sum[used]).sum()/p.size)


def compute_point_metrics(probs,target,surface=True,n_bins=15):
    if probs.shape!=target.shape:raise ValueError('Probability/target shape mismatch')
    p=probs.detach().float().cpu().numpy()
    y=target.detach().float().cpu().numpy()
    pred=p>=np.float32(.5);truth=y.astype(bool,copy=False)
    tp=np.float32(np.count_nonzero(pred & truth))
    n_pred=np.float32(np.count_nonzero(pred));n_true=np.float32(np.count_nonzero(truth))
    fp=np.float32(n_pred-tp);fn=np.float32(n_true-tp);eps=np.float32(1e-6)
    denom=np.float32(n_pred+n_true);union=np.float32(tp+fp+fn)
    dice=np.float32(1.) if denom==0 else np.float32((np.float32(2.)*tp+eps)/(denom+eps))
    iou=np.float32(1.) if union==0 else np.float32((tp+eps)/(union+eps))
    precision=np.float32((tp+eps)/(tp+fp+eps));recall=np.float32((tp+eps)/(tp+fn+eps))
    f1=np.float32(np.float32(2.)*precision*recall/(precision+recall+eps))
    hd95,assd=_surface_pair(pred,truth) if surface else (float('nan'),float('nan'))
    pf=p.reshape(-1);yf=y.reshape(-1)
    brier=float(np.mean((pf-yf)**2,dtype=np.float32))
    log_eps=np.float32(max(1e-8,torch.finfo(torch.float32).eps))
    clamped=np.clip(pf,log_eps,np.float32(1.)-log_eps)
    nll=float(np.mean(-(yf*np.log(clamped)+(np.float32(1.)-yf)*np.log1p(-clamped)),dtype=np.float32))
    seg=SegmentationMetrics(float(dice),float(iou),float(precision),float(recall),float(f1),hd95,assd)
    cal=CalibrationMetrics(_hard_ece(pf,yf,n_bins),brier,nll)
    return seg,cal,bool(n_pred==0)
