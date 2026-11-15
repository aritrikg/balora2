"""Exercise the real factory across notebook dependency-install boundaries.

The official package is a temporary tiny API stub in these CPU regressions.
Neither the production wrapper nor training.build_model is mocked.
"""
from pathlib import Path
import os
import subprocess
import sys
import textwrap

import pytest


STARTUP_SCENARIO = r'''
import importlib, pathlib, sys, tempfile
sys.modules['sam2'] = None  # Simulate the package missing before setup.
from evaluation.validation_audit import locate_results
import models.sam2_wrapper as wrapper
from training.train import build_model
cfg = {
    'sam2': {'model_cfg':'fake', 'checkpoint':'fake'},
    'lora': {'target_modules':['q_proj','out_proj'], 'rank_bank':[1,2],
             'alpha':2.0, 'dropout':0.0, 'encoder_target_modules':['qkv']},
    'bayesian': {'prior_std':1.0,'init_posterior_std':0.15},
    'training': {'amp':False},
}
if sys.argv[1] == 'after_failed_construction':
    try:
        build_model(cfg, 'cpu')
    except ImportError:
        pass
    else:
        raise AssertionError('Missing runtime dependency was silently accepted')
with tempfile.TemporaryDirectory() as folder:
    package = pathlib.Path(folder)/'sam2'
    package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'build_sam.py').write_text(
        'from tests.fake_sam2 import FakeSAM2\n'
        'def build_sam2(config, checkpoint, device, mode):\n'
        '    assert (config,checkpoint,device,mode)==("fake","fake","cpu","train")\n'
        '    return FakeSAM2(image_size=64)\n')
    del sys.modules['sam2']
    sys.path.insert(0, folder)
    # Do not reload BALoRA. This is the notebook's previously failing order.
    model = build_model(cfg, 'cpu')
    assert model.input_size == 64
    assert model.encoder_lora_layers and model.decoder_lora_layers
    assert wrapper._SAM2_IMPORT_ERROR is None
    # A later source/builder replacement must not use an old cached function.
    module = importlib.import_module('sam2.build_sam')
    calls = []
    def replacement(*args, **kwargs):
        calls.append(True)
        from tests.fake_sam2 import FakeSAM2
        return FakeSAM2(image_size=64)
    module.build_sam2 = replacement
    build_model(cfg, 'cpu')
    assert calls == [True]
print('RUNTIME_IMPORT_RECOVERY_PASS')
'''


def run_scenario(script, *args):
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = source+os.pathsep+env.get('PYTHONPATH','')
    return subprocess.run([sys.executable, '-c', textwrap.dedent(script), *args],
                          capture_output=True, text=True, env=env, timeout=90)


@pytest.mark.parametrize('scenario', ['after_audit_import', 'after_failed_construction'])
def test_model_factory_resolves_sam2_installed_later_without_reloading_balora(scenario):
    result = run_scenario(STARTUP_SCENARIO, scenario)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'RUNTIME_IMPORT_RECOVERY_PASS' in result.stdout


def test_metadata_import_never_attempts_to_import_the_optional_sam2_package():
    result = run_scenario(r'''
import builtins
original = builtins.__import__
calls = []
def track(name, *args, **kwargs):
    if name == 'sam2' or name.startswith('sam2.'):
        calls.append(name)
        raise ModuleNotFoundError('simulated not installed yet')
    return original(name, *args, **kwargs)
builtins.__import__ = track
from evaluation.validation_audit import locate_results, prepare_reference
assert calls == [], calls
print('METADATA_IMPORT_PASS')
''')
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'METADATA_IMPORT_PASS' in result.stdout
