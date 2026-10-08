"""SFT controller contracts; subprocess probes never claim model training."""
import json
from pathlib import Path
import sys

import pytest

from xqgeneral import sft
from xqgeneral.evidence import atomic_json, manifest, write_jsonl


def selected_fixture(tmp_path):
    # A byte-bound legacy parent suffices to exercise the controller, not GPU loading.
    parent = tmp_path / 'selected';parent.mkdir()
    checkpoint = parent / 'adapter.pt';checkpoint.write_bytes(b'controlled selected bytes')
    saved = {'model_path': 'controlled-model', 'model_revision': 'controlled-revision',
        'mode': 'bridge', 'decoder_bridge_positions': [0, 2], 'bridge_width': 6,
        'board_tokens': True, 'expert_feature_depths': [0, 1], 'lora_rank': 16,
        'decoder_training': 'frozen'}
    atomic_json(parent / 'config.json', saved)
    atomic_json(parent / 'manifest.json', manifest('controlled_selected_parent', {}, [], [checkpoint], {}))
    data = tmp_path / 'data'
    write_jsonl(data / 'train.jsonl', [{'stage': 'explanation'} for _ in range(18)] +
                [{'stage': 'replay'} for _ in range(6)])
    recipe = {'data_path': str(data), 'stages': ['explanation'],
        'mixture': {'explanation': .5, 'replay': .5}, 'epochs': 4, 'max_steps': 64,
        'min_steps': 64, 'batch_size': 8, 'micro_batch_size': 1, 'decoder_training': 'full'}
    return checkpoint, saved, recipe


@pytest.mark.parametrize('world', [1, 2, 4])
@pytest.mark.parametrize('resuming', [False, True])
def test_guarded_controller_launches_requested_workers_and_same_resume_path(tmp_path, monkeypatch, world, resuming):
    checkpoint, _, recipe = selected_fixture(tmp_path);recipe['ddp_world_size'] = world
    recipe_path = tmp_path / 'recipe.json';atomic_json(recipe_path, recipe)
    output = tmp_path / 'sft with spaces'
    if resuming:
        output.mkdir();(output / 'latest.pt').write_bytes(b'controlled resume bytes')
    calls = [];monkeypatch.setattr(sft.subprocess, 'run', lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setattr(sys, 'argv', ['sft', '--recipe', str(recipe_path), '--init', str(checkpoint), '--output', str(output)])
    sft.main()
    prepared = output.parent / (output.name + '.config.json');config = json.loads(prepared.read_text())
    assert config['steps'] == config['min_steps'] == 18
    assert config['ddp_world_size'] == world and config['batch_size'] == 8
    assert config['decoder_training'] == 'full' and config['model_revision'] == 'controlled-revision'
    assert len(calls) == 1 and calls[0][1] == {'check': True}
    command = calls[0][0]
    if world == 1:
        assert command[:3] == [sys.executable, '-m', 'xqgeneral.train']
    else:
        assert command[:10] == [sys.executable, '-u', '-m', 'torch.distributed.run',
            '--standalone', '--nnodes=1', f'--nproc_per_node={world}', '--module', 'xqgeneral.train', '--config']
    assert command[command.index('--config') + 1] == str(prepared)
    assert ('--resume' in command) is resuming
    if resuming:assert command[-2:] == ['--resume', str(output / 'latest.pt')]


def test_prepare_only_verifies_parent_and_writes_identical_config_without_workers(tmp_path, monkeypatch):
    checkpoint, _, recipe = selected_fixture(tmp_path);recipe['ddp_world_size'] = 4
    recipe_path = tmp_path / 'recipe.json';atomic_json(recipe_path, recipe)
    output = tmp_path / 'sft';calls = []
    monkeypatch.setattr(sft.subprocess, 'run', lambda command, **kw: calls.append(command))
    argv = ['sft', '--recipe', str(recipe_path), '--init', str(checkpoint), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv + ['--prepare-only']);sft.main()
    path = tmp_path / 'sft.config.json';prepared_bytes = path.read_bytes()
    assert not calls and not output.exists()
    monkeypatch.setattr(sys, 'argv', argv);sft.main()
    assert len(calls) == 1 and path.read_bytes() == prepared_bytes


@pytest.mark.parametrize('change', [
    {'ddp_world_size': 0}, {'ddp_world_size': True}, {'ddp_world_size': 4, 'batch_size': 10},
    {'batch_size': False}, {'micro_batch_size': 0}, {'micro_batch_size': True},
    {'epochs': 0}, {'epochs': float('nan')}, {'max_steps': 0}, {'min_steps': -1},
    {'mixture': {'explanation': 0}}, {'mixture': {'explanation': 1.1}},
])
def test_invalid_sft_budget_refused_before_config_write_or_launch(tmp_path, monkeypatch, change):
    checkpoint, _, recipe = selected_fixture(tmp_path);recipe.update(change)
    recipe_path = tmp_path / 'recipe.json';atomic_json(recipe_path, recipe)
    calls = [];monkeypatch.setattr(sft.subprocess, 'run', lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(sys, 'argv', ['sft', '--recipe', str(recipe_path), '--init', str(checkpoint),
        '--output', str(tmp_path / 'sft'), '--prepare-only'])
    with pytest.raises(ValueError):sft.main()
    assert not calls and not (tmp_path / 'sft.config.json').exists()


def test_prepared_configuration_cannot_be_overwritten_by_changed_budget(tmp_path, monkeypatch):
    checkpoint, _, recipe = selected_fixture(tmp_path);recipe_path = tmp_path / 'recipe.json'
    atomic_json(recipe_path, recipe)
    monkeypatch.setattr(sys, 'argv', ['sft', '--recipe', str(recipe_path), '--init', str(checkpoint),
        '--output', str(tmp_path / 'sft'), '--prepare-only'])
    sft.main();path = tmp_path / 'sft.config.json';original = path.read_bytes()
    recipe['epochs'] = 5;atomic_json(recipe_path, recipe)
    with pytest.raises(ValueError, match='Prepared SFT configuration differs'):sft.main()
    assert path.read_bytes() == original


@pytest.mark.parametrize('problem', ['incomplete', 'changed_checkpoint'])
def test_prepare_only_keeps_completed_parent_guard(tmp_path, monkeypatch, problem):
    checkpoint, _, recipe = selected_fixture(tmp_path);recipe_path = tmp_path / 'recipe.json'
    atomic_json(recipe_path, recipe)
    if problem == 'incomplete':
        path = checkpoint.parent / 'manifest.json';proof = json.loads(path.read_text())
        proof['status'] = 'incomplete';atomic_json(path, proof)
    else:checkpoint.write_bytes(b'changed')
    monkeypatch.setattr(sys, 'argv', ['sft', '--recipe', str(recipe_path), '--init', str(checkpoint),
        '--output', str(tmp_path / 'sft'), '--prepare-only'])
    with pytest.raises(ValueError, match='completed, hash-verified'):sft.main()
    assert not (tmp_path / 'sft.config.json').exists()


def test_budget_reads_primary_training_only_and_honors_cap(tmp_path):
    checkpoint, saved, recipe = selected_fixture(tmp_path)
    for split in ['validation', 'test']:
        (Path(recipe['data_path']) / (split + '.jsonl')).write_text('Held-out contents must not enter budget calculation')
    recipe['max_steps'] = 12
    config = sft.prepare_config(saved, recipe, checkpoint, tmp_path / 'sft')
    assert config['steps'] == config['min_steps'] == 12
    assert config['decoder_bridge_positions'] == saved['decoder_bridge_positions']


@pytest.mark.parametrize('prepare_only', [False, True])
def test_clean_recipe_rejects_legacy_parent_before_config_or_worker(tmp_path, monkeypatch, prepare_only):
    checkpoint, _, recipe = selected_fixture(tmp_path)
    recipe['require_clean_foundation_handoff'] = True
    recipe_path = tmp_path / 'recipe.json';atomic_json(recipe_path, recipe)
    calls = [];monkeypatch.setattr(sft.subprocess, 'run', lambda *a, **kw: calls.append(a))
    argv = ['sft', '--recipe', str(recipe_path), '--init', str(checkpoint), '--output', str(tmp_path / 'sft')]
    monkeypatch.setattr(sys, 'argv', argv + (['--prepare-only'] if prepare_only else []))
    with pytest.raises(ValueError, match='completed four-course clean foundation handoff'):sft.main()
    assert not calls and not (tmp_path / 'sft.config.json').exists()
