"""Reuse a measured control profile without repeating discarded updates."""
from __future__ import annotations
import copy,json,math,re
from pathlib import Path
from training.deterministic_control import RUNTIME_REVISION
from utils.runtime_budget import write_json


def validate_profile(profile,cfg):
    timings=profile.get('training_sample_seconds',[])
    cases=profile.get('training_sample_cases',[])
    if (not profile.get('complete') or profile.get('runtime_revision')!=RUNTIME_REVISION
            or profile.get('protocol_sha256')!=cfg['control']['protocol_sha256']):
        raise ValueError('Measured profile belongs to another protocol/runtime')
    if (not timings or len(timings)!=len(cases)
            or set(cases)!=set(cfg['experiment']['split_case_ids']['train'])):
        raise ValueError('Measured profile does not cover the declared training cases')
    for value in [*timings,profile.get('validation_sample_seconds',float('nan')),
                  profile.get('training_p75_seconds_per_slice',float('nan'))]:
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Profile durations must be finite and positive')
    expected=sorted(timings)[math.ceil(.75*len(timings))-1]
    if not math.isclose(expected,profile['training_p75_seconds_per_slice'],abs_tol=1e-9):
        raise ValueError('Profile p75 does not match its measured samples')
    for name in ('train_foreground_slices','validation_foreground_slices','validation_sample_slices'):
        if not isinstance(profile.get(name),int) or profile[name]<=0:
            raise ValueError('Invalid measured foreground slice counts')
    if cfg['data']['num_classes']==14 and (profile['train_foreground_slices']!=1347 or profile['validation_foreground_slices']!=371):
        raise ValueError('Measured profile differs from original Fold-1 coverage')
    if not isinstance(profile.get('gpu'),str) or not profile['gpu']:
        raise ValueError('Measured profile must identify its device')
    return profile


def stage_saved_profile(root,cfg,record,prior_accounted_seconds):
    """Stage prior timings; profile_control still requires the same GPU name."""
    if not re.fullmatch('[0-9a-f]{64}',record.get('source_archive_sha256','')):
        raise ValueError('Source archive digest is required')
    used=float(record['accounted_runtime_seconds'])
    if not math.isfinite(used) or used<0 or prior_accounted_seconds+1e-6<used:
        raise ValueError('Prior accounting must include the measured profile run')
    profile=validate_profile(copy.deepcopy(record['profile']),cfg)
    path=Path(root)/'profile_report.json'
    if path.is_file():
        current=json.loads(path.read_text())
        try:
            validate_profile(current,cfg)
        except ValueError:
            pass
        else:
            return {'staged':False,'source':'attached/local compatible profile','gpu':current['gpu']}
    profile['cache_provenance']={k:record[k] for k in
        ('source_archive_sha256','accounted_runtime_seconds','source_execution_status')}
    write_json(path,profile)
    return {'staged':True,'source':'embedded measured profile from latest supplied ZIP','gpu':profile['gpu']}
