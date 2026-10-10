"""Controlled artifact integration: no real GPU model or student training."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_foundation_handoff import completed_curriculum
from xqgeneral import modeling, sft_pipeline, sft_preflight, sft_probe_worker
from xqgeneral.evidence import atomic_json, code_identity, digest, manifest, write_jsonl
from xqgeneral.foundation_handoff import export_curriculum
from xqgeneral.sft import prepare_config


def setup_data(tmp_path):
    curriculum, _ = completed_curriculum(tmp_path)
    foundation = json.loads((curriculum / 'manifest.json').read_text())['config']
    weights = Path(foundation['model_path']) / 'model.safetensors';weights.write_bytes(b'controlled base identity')
    data = tmp_path / 'labels'
    lengths = []
    for split, count in [('train', 32), ('validation', 16)]:
        rows = [{'id': split + str(i).zfill(3), 'split': split, 'stage': 'explanation', 'answer': 'preserved answer'}
                for i in range(count)]
        write_jsonl(data / (split + '.jsonl'), rows)
        lengths += [{'id': row['id'], 'split': split, 'tokens': 100 + i} for i, row in enumerate(rows)]
    token_root = tmp_path / 'token-preflight'
    write_jsonl(token_root / 'token-lengths.jsonl', lengths)
    foundation_path = tmp_path / 'foundation-preflight.json'
    atomic_json(foundation_path, manifest('recorded_engine_clean_latent_foundation_preflight', {}, [weights], [], {}))
    token_path = token_root / 'manifest.json'
    token = manifest('latent_only_reviewed_explanation_data_preflight',
        {'data': str(data), 'foundation_preflight': str(foundation_path)},
        [data / 'train.jsonl', data / 'validation.jsonl', foundation_path],
        [token_root / 'token-lengths.jsonl'],
        {'token_limit': 1024, 'all_train_validation_supervision_fits_without_truncation': True})
    cache = Path(foundation['feature_path'])
    token['cache_header_only_input'] = {'path': str(cache), 'sha256': digest(cache), 'bytes': cache.stat().st_size}
    atomic_json(token_path, token)
    recipe = json.loads(Path('configs/explanation-sft-clean-v1.json').read_text())
    recipe.update(data_path=str(data), feature_path=str(cache))
    recipe_path = tmp_path / 'sft-recipe.json';atomic_json(recipe_path, recipe)
    return curriculum, recipe_path, token_path


class ControlledFullModel(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weights = {name: torch.nn.Parameter(torch.zeros_like(value)) for name, value in source.items()}
        self.weights['base.decoder.weight'] = torch.nn.Parameter(torch.ones(2, 3))

    def named_parameters(self, *args, **kwargs):return iter(self.weights.items())


class ControlledRunner:
    def __init__(self, original):self.original = original;self.jobs = [];self.failure = None

    def __call__(self, command, **kwargs):
        if command[0] == 'git':return self.original(command, **kwargs)
        if command[0] == 'tmux':return SimpleNamespace(returncode=0)
        self.jobs.append(command)
        if self.failure == 'preflight' and 'xqgeneral.sft_probe_worker' in command:
            raise subprocess.CalledProcessError(1, command)
        if 'xqgeneral.sft_probe_worker' in command:
            config = json.loads(Path(command[command.index('--config') + 1]).read_text())
            root = Path(config['output']);root.mkdir(parents=True, exist_ok=True)
            step = int(command[command.index('--stop-after') + 1])
            initial = torch.load(root.parent / 'zero-init.pt', weights_only=True)['trainable']
            state = {n: t + step * 1e-4 for n, t in initial.items()}
            ids = {name: i for i, name in enumerate(initial)}
            groups = []
            for group in ['bridge', 'board', 'decoder']:
                names = [n for n in initial if ('board' if n.endswith('board_weight') else
                         ('bridge' if n.startswith('bridges.') else 'decoder')) == group]
                groups.append({'name': group, 'params': [ids[n] for n in names]})
            saved = {'step': step, 'world_size': 4, 'trainable': state, 'config': config,
                'optimizer': {'state': {ids[n]: {'step': torch.tensor(float(step)),
                    'exp_avg': torch.ones_like(t) * step, 'exp_avg_sq': torch.ones_like(t) * step}
                    for n, t in initial.items()}, 'param_groups': groups},
                'random_state': random.Random(8).getstate(), 'torch_rng_state': torch.tensor([8], dtype=torch.uint8),
                'cuda_rng_states': [torch.tensor([rank], dtype=torch.uint8) for rank in range(4)],
                'tokens_seen': step * 100, 'initial_loss': 3.0, 'code': code_identity(),
                'compute_contract': {'deterministic_training': True}, 'patience_count': 0,
                'best_step': step, 'best_loss': 1 / step,
                'input_hashes': {str(p): digest(p) for p in [Path(config['feature_path']),
                    Path(config['data_path']) / 'train.jsonl', Path(config['data_path']) / 'validation.jsonl',
                    Path(config['init_from'])]}}
            torch.save(saved, root / 'latest.pt');atomic_json(root / 'config.json', config)
            atomic_json(root / 'execution.json', {'code': saved['code'], 'input_hashes': saved['input_hashes']})
            write_jsonl(root / 'training.jsonl', [{'step': s, 'loss': 1 / s, 'gradient_norm': .5,
                         'seconds': s} for s in range(1, step + 1)])
            for rank in range(4):
                atomic_json(root / f'resources-step-{step}-rank-{rank}.json', {'rank': rank,
                    'world_size': 4, 'actual_step': step, 'peak_allocated_gib': .25,
                    'total_memory_gib': 96.0, 'training_main_returned_successfully': True})
            if self.failure == 'changed_probe_cache':
                p = Path(config['feature_path']);p.write_bytes(p.read_bytes() + b'changed')
        elif 'xqgeneral.sft' in command:
            source = Path(command[command.index('--init') + 1]);output = Path(command[command.index('--output') + 1])
            recipe = json.loads(Path(command[command.index('--recipe') + 1]).read_text())
            saved = json.loads((source.parent / 'config.json').read_text())
            config = prepare_config(saved, recipe, source, output);output.mkdir(parents=True)
            initial = torch.load(output.parent / 'preflight/zero-init.pt', weights_only=True)['trainable']
            state = {n: t + .01 for n, t in initial.items()}
            if self.failure == 'selected_state':state['base.decoder.weight'] = torch.ones(99)
            inputs = [Path(config['data_path']) / 'train.jsonl', Path(config['data_path']) / 'validation.jsonl', source]
            actual_inputs = {str(p): digest(p) for p in [Path(config['feature_path']), *inputs]}
            checkpoint = {'trainable': state, 'config': config, 'selected_step': config['steps'],
                          'code': code_identity(), 'input_hashes': actual_inputs}
            torch.save(checkpoint, output / 'adapter.pt');atomic_json(output / 'config.json', config)
            metrics = {'world_size': 4, 'decoder_training': 'full', 'selected_step': config['steps'],
                'steps_completed': config['steps'], 'global_batch_size': 16,
                'trainable_parameters': sum(t.numel() for t in initial.values())}
            atomic_json(output / 'metrics.json', metrics)
            atomic_json(output / 'manifest.json', manifest('model_training', config, inputs,
                [output / 'adapter.pt'], metrics))
            if self.failure == 'changed_formal_labels':
                p = Path(config['data_path']) / 'train.jsonl';p.write_bytes(p.read_bytes() + b'changed')
        else:raise AssertionError('Unexpected controlled command: ' + str(command))
        return SimpleNamespace(returncode=0)


def install_runner(monkeypatch):
    runner = ControlledRunner(subprocess.run)
    monkeypatch.setattr(subprocess, 'run', runner)
    def load(config, **kwargs):
        source = torch.load(config['init_from'], weights_only=True)['trainable']
        return ControlledFullModel(source), None
    monkeypatch.setattr(modeling, 'load_model', load)
    return runner


def test_sequential_pipeline_controlled_complete_handoff_probe_and_formal_sft(tmp_path, monkeypatch):
    curriculum, recipe, token = setup_data(tmp_path);runner = install_runner(monkeypatch)
    output = tmp_path / 'pipeline'
    result = sft_pipeline.run_pipeline(curriculum, 'owned-session', recipe, token, output)
    assert result['initial_explanation_sft_completed'] and result['actual_sft_steps'] == 8
    assert result['selected_sft_step'] == 8 and result['complete_shapes_and_finite_fp32_verified']
    assert not result['strong_play_or_reliable_coaching_proven']
    assert len(runner.jobs) == 4
    assert all('xqgeneral.sft_probe_worker' in command for command in runner.jobs[:3])
    assert [int(c[c.index('--stop-after') + 1]) for c in runner.jobs[:3]] == [4, 2, 4]
    assert '--resume' not in runner.jobs[1] and '--resume' in runner.jobs[2]
    assert 'xqgeneral.sft' in runner.jobs[3]
    assert result == sft_pipeline.run_pipeline(curriculum, 'owned-session', recipe, token, output, resume=True)
    assert len(runner.jobs) == 4
    selected = json.loads((output / 'preflight/verification.json').read_text())['longest_actual_train_validation_probe']
    assert selected['train']['ids'][0] == 'train031' and len(selected['train']['ids']) == 24
    assert selected['validation']['ids'][0] == 'validation015'


def test_missing_producer_does_not_restart_or_load_gpu(tmp_path, monkeypatch):
    curriculum, recipe, token = setup_data(tmp_path)
    (curriculum / 'manifest.json').rename(curriculum / 'preserved-four-course.json')
    runner = install_runner(monkeypatch)
    original = runner.__call__
    def missing(command, **kwargs):
        if command[0] == 'tmux':return SimpleNamespace(returncode=1)
        return original(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', missing)
    output = tmp_path / 'waiting'
    with pytest.raises(RuntimeError, match='no longer live'):
        sft_pipeline.run_pipeline(curriculum, 'owned-session', recipe, token, output)
    assert not runner.jobs and not (output / 'handoff').exists()
    assert json.loads((output / 'state.json').read_text())['stage'] == 'waiting_for_four_courses'


def test_live_wait_uses_same_producer_until_actual_complete_manifest(tmp_path, monkeypatch):
    curriculum, _, _ = setup_data(tmp_path);path = curriculum / 'manifest.json';original = path.read_bytes();path.unlink()
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0))
    output = tmp_path / 'wait';output.mkdir()
    def advance(seconds):
        assert json.loads((output / 'state.json').read_text())['gpu_model_loaded_or_sft_started'] is False
        assert seconds == 1;path.write_bytes(original)
    monkeypatch.setattr(sft_pipeline.time, 'sleep', advance)
    assert sft_pipeline.wait_for_curriculum(curriculum, 'owned-session', output, 1) == path
    assert calls == [['tmux', 'has-session', '-t', 'owned-session']]


@pytest.mark.parametrize('failure', ['preflight', 'selected_state', 'changed_probe_cache', 'changed_formal_labels'])
def test_actual_phase_failure_preserves_evidence_and_never_claims_completion(tmp_path, monkeypatch, failure):
    curriculum, recipe, token = setup_data(tmp_path);runner = install_runner(monkeypatch);runner.failure = failure
    output = tmp_path / 'failed'
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        sft_pipeline.run_pipeline(curriculum, 'owned-session', recipe, token, output)
    assert json.loads((output / 'state.json').read_text())['status'] == 'failed'
    assert not (output / 'manifest.json').exists()
    if failure in ['preflight', 'changed_probe_cache']:
        assert not any('xqgeneral.sft' in c for c in runner.jobs)


def test_changed_waiting_recipe_rejected_before_handoff_or_gpu(tmp_path, monkeypatch):
    curriculum, recipe, token = setup_data(tmp_path);runner = install_runner(monkeypatch)
    path = curriculum / 'manifest.json';saved = path.read_bytes();path.unlink()
    def changed(seconds):
        r=json.loads(recipe.read_text());r['epochs']=5;atomic_json(recipe,r);path.write_bytes(saved)
    monkeypatch.setattr(sft_pipeline.time, 'sleep', changed)
    output = tmp_path / 'changed'
    with pytest.raises(ValueError, match='Pipeline recipe'):
        sft_pipeline.run_pipeline(curriculum, 'owned-session', recipe, token, output, poll_seconds=1)
    assert not runner.jobs and not (output / 'handoff').exists()


@pytest.mark.parametrize('problem', ['wrong_kind', 'changed_labels', 'wrong_token_limit', 'changed_feature_path'])
def test_preflight_rejects_wrong_completed_data_before_workers(tmp_path, monkeypatch, problem):
    curriculum, recipe, token = setup_data(tmp_path);runner = install_runner(monkeypatch)
    handoff = tmp_path / 'handoff';export_curriculum(curriculum, handoff)
    if problem == 'changed_labels':
        p=Path(json.loads(recipe.read_text())['data_path'])/'train.jsonl';p.write_bytes(p.read_bytes()+b'changed')
    else:
        d=json.loads(token.read_text())
        if problem == 'wrong_kind':d['kind']='unrelated'
        elif problem == 'wrong_token_limit':d['verification']['token_limit']=32
        else:d['cache_header_only_input']['path']='unrelated-feature-cache.pt'
        atomic_json(token,d)
    with pytest.raises(ValueError):
        sft_preflight.run_preflight(handoff/'adapter.pt', recipe, token, tmp_path/'probe')
    assert not runner.jobs and not (tmp_path/'probe').exists()


def test_preflight_resume_requires_unchanged_recipe_and_executed_source(tmp_path, monkeypatch):
    curriculum, recipe, token = setup_data(tmp_path);runner=install_runner(monkeypatch)
    handoff=tmp_path/'handoff';export_curriculum(curriculum,handoff)
    probe=tmp_path/'probe';sft_preflight.run_preflight(handoff/'adapter.pt',recipe,token,probe)
    count=len(runner.jobs)
    r=json.loads(recipe.read_text());r['epochs']=5;atomic_json(recipe,r)
    with pytest.raises(ValueError, match='exact immutable contract'):
        sft_preflight.run_preflight(handoff/'adapter.pt',recipe,token,probe,resume=True)
    assert len(runner.jobs)==count


@pytest.mark.parametrize('problem', ['moments', 'group_shape', 'zero_gradient', 'missing_resource', 'wrong_code', 'changed_config', 'resume_mismatch', 'missing_rng'])
def test_self_consistent_short_runs_cannot_hide_invalid_actual_probe(tmp_path, monkeypatch, problem):
    curriculum, recipe, token = setup_data(tmp_path);install_runner(monkeypatch)
    handoff=tmp_path/'handoff';export_curriculum(curriculum,handoff)
    probe=tmp_path/'probe';sft_preflight.run_preflight(handoff/'adapter.pt',recipe,token,probe)
    expected=torch.load(probe/'continuous/latest.pt',weights_only=True)
    config=deepcopy(expected['config']);inputs=deepcopy(expected['input_hashes']);code=deepcopy(expected['code'])
    if problem == 'resume_mismatch':
        path=probe/'resumed/latest.pt';d=torch.load(path,weights_only=True)
        d['trainable']['base.decoder.weight']=torch.ones(99);torch.save(d,path)
    elif problem in ['moments','group_shape','wrong_code','changed_config','missing_rng']:
        for run in ['continuous','resumed']:
            path=probe/run/'latest.pt';d=torch.load(path,weights_only=True)
            if problem=='moments':d['optimizer']['state'][0]['exp_avg'].fill_(float('inf'))
            elif problem=='group_shape':d['optimizer']['state'][0]['exp_avg']=torch.ones(99)
            elif problem=='wrong_code':d['code']={'revision':'unrelated','source_sha256':{}}
            elif problem=='changed_config':d['config']['batch_size']=32
            else:d.pop('torch_rng_state')
            torch.save(d,path)
    elif problem=='zero_gradient':
        for run in ['continuous','resumed']:
            path=probe/run/'training.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
            rows[0]['gradient_norm']=0;write_jsonl(path,rows)
    else:(probe/'continuous/resources-step-4-rank-3.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        sft_preflight.verify_updated_probe(probe,probe/'zero-init.pt',inputs,config,code)


def test_probe_worker_records_actual_rank_and_step_after_training_returns(tmp_path, monkeypatch):
    root=tmp_path/'worker';root.mkdir();write_jsonl(root/'training.jsonl',[{'step':4}])
    config=tmp_path/'worker.json';atomic_json(config,{'output':str(root)})
    calls=[];monkeypatch.setattr(sft_probe_worker.training,'main',lambda:calls.append('returned'))
    monkeypatch.setenv('RANK','2');monkeypatch.setenv('WORLD_SIZE','4');monkeypatch.setenv('LOCAL_RANK','2')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda device:72*2**30)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(total_memory=96*2**30))
    monkeypatch.setattr(sys,'argv',['worker','--config',str(config),'--stop-after','4'])
    sft_probe_worker.main();r=json.loads((root/'resources-step-4-rank-2.json').read_text())
    assert calls==['returned'] and r['rank']==2 and r['world_size']==4 and r['actual_step']==4
    assert r['peak_allocated_gib']==72 and r['total_memory_gib']==96


@pytest.mark.parametrize('problem',['missing','shape','nonfinite','precision'])
def test_final_full_parameter_contract_rejects_incomplete_or_invalid_selected_state(problem):
    initial={'base.decoder.weight':torch.ones(2,3),'bridges.0.weight':torch.ones(2,3)}
    state={n:t.clone() for n,t in initial.items()}
    if problem=='missing':state.pop('base.decoder.weight')
    elif problem=='shape':state['base.decoder.weight']=torch.ones(99)
    elif problem=='nonfinite':state['base.decoder.weight'].fill_(float('inf'))
    else:state['base.decoder.weight']=state['base.decoder.weight'].half()
    with pytest.raises(ValueError):sft_preflight.full_parameter_summary(state,initial)
