"""Exercise the real CLI's first digest in a fresh OS process, before models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from xqgeneral.evidence import digest, file_signature
from xqgeneral.verified_inputs import ENVIRONMENT, write_verified_input_receipt


CHILD = r'''
import importlib, json, sys
from pathlib import Path
from xqgeneral import evidence, recorded_search_inputs
from xqgeneral.verified_inputs import input_digest_execution

name, asset = sys.argv[1], Path(sys.argv[2]).resolve()
module = importlib.import_module('xqgeneral.' + name)
original_open, original_digest = Path.open, evidence.digest
reads, queries = [], []
result = {}

class StopProbe(BaseException):
    pass

def opened(path, mode='r', *args, **kwargs):
    if Path(path).resolve() == asset and mode == 'rb':
        reads.append(str(path))
    return original_open(path, mode, *args, **kwargs)

def first_digest(path):
    if Path(path).resolve() != asset:
        return original_digest(path)
    result.update(digest=original_digest(path), session=input_digest_execution())
    raise StopProbe

def refuse_model(*args, **kwargs):
    queries.append('unexpected model or engine construction')
    raise AssertionError(queries[-1])

Path.open = opened
if hasattr(module, 'digest'):
    module.digest = first_digest
recorded_search_inputs.digest = first_digest
if hasattr(module, 'Pikafish'):
    module.Pikafish = refuse_model
if name == 'search_distillation':
    from xqgeneral import inference
    inference.Predictor = refuse_model
if name == 'local_teacher':
    module.LocalConsolidator = refuse_model
sys.argv = [name, *sys.argv[3:]]
try:
    module.main()
except StopProbe:
    result['stopped_at_actual_native_digest'] = True
except Exception as error:
    result.update(error_type=type(error).__name__, error=str(error))
else:
    result['returned_without_reaching_native_digest'] = True
import torch
result.update(physical_binary_reads=len(reads), model_or_engine_loads=len(queries),
              CUDA_initialized=torch.cuda.is_initialized())
print(json.dumps(result))
'''


def register(asset, receipt):
    write_verified_input_receipt(receipt, {str(asset): {
        'sha256': digest(asset), 'bytes': asset.stat().st_size,
        'signature': list(file_signature(asset))}})


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp('search-cli-session')
    asset = root / 'verified.pt'
    asset.write_text(json.dumps({'id': 'cpu-query', 'messages': [{'role': 'user', 'content': 'CPU control'}]}) + '\n')
    receipt = root / 'receipt.json'
    register(asset, receipt)
    config = root / 'teacher.json'
    config.write_text(json.dumps({'search_consolidator': {'model_path': str(root / 'unused-model')}}))
    return asset, receipt, config


@pytest.mark.parametrize('module', ['search_pilot', 'search_distillation', 'local_teacher', 'collect_search'])
@pytest.mark.parametrize('receipt_mode', ['verified', 'absent', 'changed'])
def test_real_search_cli_process_scopes_verified_inputs(module, receipt_mode, inputs, tmp_path):
    asset, receipt, config = inputs
    if receipt_mode == 'changed':
        asset, receipt = tmp_path / 'changed.pt', tmp_path / 'stale-receipt.json'
        asset.write_bytes(inputs[0].read_bytes())
        register(asset, receipt)
        asset.write_bytes(asset.read_bytes().replace(b'CPU control', b'new control'))
    output = str(tmp_path / 'unused-output')
    arguments = {
        'search_pilot': ['--pipeline', 'unused', '--validation', 'unused', '--producer-session', 'cpu',
                         '--config', str(asset), '--output', output],
        'search_distillation': ['--checkpoint', str(asset), '--features', str(asset), '--output', output],
        'local_teacher': ['--config', str(config), '--input', str(asset), '--output', output],
        'collect_search': ['--queries', str(asset), '--responses', 'unused', '--output', output],
    }
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    env.pop(ENVIRONMENT, None)
    if receipt_mode != 'absent':
        env[ENVIRONMENT] = str(receipt)
    child = subprocess.run([sys.executable, '-c', CHILD, module, str(asset), *arguments[module]],
                           cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result['model_or_engine_loads'] == 0 and result['CUDA_initialized'] is False
    if receipt_mode == 'changed':
        assert result.get('error_type') == 'ValueError', result
        assert 'identity changed' in result['error'] and result['physical_binary_reads'] == 0
    else:
        assert result.get('stopped_at_actual_native_digest') is True, result
        assert result['digest'] == digest(asset)
        if receipt_mode == 'verified':
            assert result['physical_binary_reads'] == 0, result
            session = result['session']['input_digest_verification']
            assert session['all_receipt_file_signatures_checked'] is True
            assert session['reused_digest_calls'] == 1 and session['fresh_digest_calls'] == 0
            assert session['reused_bytes_were_freshly_hashed_in_this_process'] is False
        else:
            assert result['physical_binary_reads'] == 1 and result['session'] == {}
