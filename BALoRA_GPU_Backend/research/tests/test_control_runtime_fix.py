import math
import numpy as np
import pytest
import torch
from evaluation.metrics import compute_segmentation_metrics,compute_calibration_metrics,expected_calibration_error
from evaluation.point_metrics import compute_point_metrics,_hard_ece
from training.train import _non_finite_gradient_names


@pytest.mark.parametrize('seed',[4,9,17])
@pytest.mark.parametrize('surface',[False,True])
def test_combined_metrics_match_original_definitions(seed,surface):
    g=torch.Generator().manual_seed(seed)
    p=torch.rand(48,64,generator=g);y=(torch.rand(48,64,generator=g)>.8).float()
    a,b,empty=compute_point_metrics(p,y,surface)
    expected_seg=compute_segmentation_metrics(p>=.5,y,surface).to_dict()
    expected_cal=compute_calibration_metrics(p,y).to_dict()
    for key,value in a.to_dict().items():
        if math.isnan(expected_seg[key]):assert math.isnan(value)
        else:assert value==expected_seg[key]
    for key,value in b.to_dict().items():
        assert value==pytest.approx(expected_cal[key],abs=2e-7,rel=2e-6)
    assert empty is False


@pytest.mark.parametrize('kind',['all_empty','pred_empty','target_empty','all_full'])
def test_empty_and_full_surface_conventions_are_preserved(kind):
    p=torch.zeros(24,24) if kind in ('all_empty','pred_empty') else torch.ones(24,24)
    y=torch.zeros(24,24) if kind in ('all_empty','target_empty') else torch.ones(24,24)
    a,b,empty=compute_point_metrics(p,y,True)
    expected=compute_segmentation_metrics(p>=.5,y,True).to_dict()
    assert a.to_dict()==expected
    assert empty==(not bool(p.any()))
    assert all(math.isfinite(v) for v in b.to_dict().values())


@pytest.mark.parametrize('bins',[2,15,31])
def test_histogram_ece_preserves_exact_bin_edge_assignment(bins):
    edges=np.linspace(0,1,bins+1).astype(np.float32)
    p=np.concatenate([edges,np.nextafter(edges,np.float32(-np.inf)),
                        np.nextafter(edges,np.float32(np.inf))])
    p=p[(p>=0)&(p<=1)]
    y=(np.arange(p.size)%3==0).astype(np.float32)
    expected=expected_calibration_error(torch.from_numpy(p),torch.from_numpy(y),bins)
    assert _hard_ece(p,y,bins)==pytest.approx(expected,abs=1e-7)


def test_native_1024_metrics_agree_without_pixel_subsampling():
    g=torch.Generator().manual_seed(812)
    p=torch.rand(1024,1024,generator=g);y=(torch.rand(1024,1024,generator=g)>.94).float()
    seg,cal,_=compute_point_metrics(p,y,False)
    expected_seg=compute_segmentation_metrics(p>=.5,y,False).to_dict()
    for k in ('dice','iou','precision','recall','f1'):assert seg.to_dict()[k]==expected_seg[k]
    expected_cal=compute_calibration_metrics(p,y).to_dict()
    for k,v in cal.to_dict().items():assert v==pytest.approx(expected_cal[k],abs=2e-7,rel=2e-6)


def test_batched_gradient_check_finds_only_nan_and_inf_and_handles_absent_grads():
    model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Linear(5,2))
    assert _non_finite_gradient_names(model)==[]
    for p in model.parameters():p.grad=torch.ones_like(p)
    assert _non_finite_gradient_names(model)==[]
    model[0].weight.grad[0,0]=float('nan')
    model[1].bias.grad[0]=float('inf')
    assert _non_finite_gradient_names(model)==['0.weight','1.bias']
    model[0].weight.grad=None
    assert _non_finite_gradient_names(model)==['1.bias']


def test_old_runtime_profile_is_remeasured_after_speed_revision(tmp_path):
    import json,time
    from tests.test_deterministic_control import cfg,model,loader
    from training.deterministic_control import profile_control,RUNTIME_REVISION
    c=cfg(tmp_path)
    (tmp_path/'profile_report.json').write_text(json.dumps({
        'complete':True,'protocol_sha256':c['control']['protocol_sha256'],
        'gpu':'CPU_TEST','training_p75_seconds_per_slice':999.}))
    p=profile_control(c,loader(['t1','t2']),loader(['v1','v2']),'cpu',
                      deadline=time.time()+60,model=model(c))
    assert p['runtime_revision']==RUNTIME_REVISION
    assert p['training_p75_seconds_per_slice']<999.
