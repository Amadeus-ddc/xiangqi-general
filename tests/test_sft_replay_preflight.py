"""Real multi-file indexes and finite state, with controlled GPU artifacts."""
import json
from pathlib import Path
import subprocess

import pytest
import torch

from test_sft_pipeline import ControlledRunner, install_runner, setup_data
from xqgeneral import sft_preflight
from xqgeneral.evidence import atomic_json, digest, iter_jsonl, write_jsonl
from xqgeneral.finite_training import FINITE_PROFILE, prepare_finite_epochs
from xqgeneral.foundation_handoff import export_curriculum
from xqgeneral.training import training_input_paths
from xqgeneral.training_index import IndexedTrainingRows


class IndexedProbeRunner(ControlledRunner):
    """Preserve the actual data contract in each controlled worker checkpoint."""
    def __init__(self, original, replay_sources):
        super().__init__(original)
        self.replay_sources = replay_sources

    def __call__(self, command, **kwargs):
        result = super().__call__(command, **kwargs)
        if 'xqgeneral.sft_probe_worker' not in command:
            return result
        config = json.loads(Path(command[command.index('--config') + 1]).read_text())
        path = Path(config['output']) / 'latest.pt'
        saved = torch.load(path, weights_only=True)
        sources = [Path(config['data_path']) / 'train.jsonl', *self.replay_sources]
        rows = IndexedTrainingRows(config['row_index_paths']['train'], sources, config['mixture'])
        try:
            saved['data_state'] = prepare_finite_epochs(rows, config['mixture'], config, 4).state_at(saved['step'])
        finally:
            rows.close()
        saved['input_hashes'] = {str(p): digest(p) for p in training_input_paths(config)}
        torch.save(saved, path)
        atomic_json(path.parent / 'execution.json', {'code': saved['code'], 'input_hashes': saved['input_hashes']})
        return result


def replay_probe(tmp_path, monkeypatch, count=1):
    curriculum, recipe_path, token_path = setup_data(tmp_path)
    recipe = json.loads(recipe_path.read_text())
    token = json.loads(token_path.read_text())
    for split in ['train', 'validation']:
        path = Path(recipe['data_path']) / (split + '.jsonl')
        rows = list(iter_jsonl(path))
        for row in rows:
            row['feature_key'] = 'fixture-history-' + row['id']
        write_jsonl(path, rows)
        token['inputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(token_path, token)
    roots = []
    for number in range(count):
        root = tmp_path / f'replay-{number}'
        for split, size in [('train', 8), ('validation', 2)]:
            write_jsonl(root / (split + '.jsonl'), [
                {'id': f'replay-{number}-{split}-{i}', 'feature_key': f'old-history-{number}-{split}-{i}',
                 'stage': f'old-{number}', 'split': split, 'answer': 'preserved old answer'}
                for i in range(size)])
        roots.append(root)
    mixture = {'explanation': .8, **{f'old-{i}': .2 / count for i in range(count)}} if count else {'explanation': 1.0}
    recipe.update(training_data_profile=FINITE_PROFILE, stages=list(mixture), mixture=mixture,
                  replay_data_paths=[str(p) for p in roots], mixture_seed=0,
                  row_index_paths={s: str(tmp_path / ('formal-' + s + '-index')) for s in ['train', 'validation']})
    atomic_json(recipe_path, recipe)
    original = install_runner(monkeypatch)
    runner = IndexedProbeRunner(original.original, [p / 'train.jsonl' for p in roots])
    monkeypatch.setattr(subprocess, 'run', runner)
    handoff = tmp_path / 'handoff'
    export_curriculum(curriculum, handoff)
    root = tmp_path / 'probe'
    result = sft_preflight.run_preflight(handoff / 'adapter.pt', recipe_path, token_path, root)
    return root, result, roots, recipe_path, token_path, handoff / 'adapter.pt', runner


@pytest.mark.parametrize('count', [0, 1, 2])
def test_indexed_probe_checks_all_configured_sources_and_real_finite_state(tmp_path, monkeypatch, count):
    root, result, roots, _, _, _, runner = replay_probe(tmp_path, monkeypatch, count)
    assert result['exact_resume']['mismatched_fields'] == []
    assert result['all_trainable_and_adam_states_finite_fp32']
    saved = torch.load(root / 'continuous/latest.pt', weights_only=True)
    assert result['actual_finite_epoch_state_verified']
    assert result['training_data_state'] == saved['data_state']
    contract = saved['data_state']['contract']
    assert contract['source_rows_by_stage']['explanation'] == 24
    assert all(contract['source_rows_by_stage'][f'old-{i}'] == 8 and
               contract['selected_rows_by_stage'][f'old-{i}'] > 0 for i in range(count))
    assert len(runner.jobs) == 3
    proof = json.loads((root / 'manifest.json').read_text())
    assert all(str(path / (split + '.jsonl')) in proof['inputs']
               for path in roots for split in ['train', 'validation'])


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_completed_replay_probe_rejects_changed_replay_source_on_resume(tmp_path, monkeypatch, split):
    root, _, roots, recipe, token, source, runner = replay_probe(tmp_path, monkeypatch)
    jobs = len(runner.jobs)
    path = roots[0] / (split + '.jsonl')
    rows = list(iter_jsonl(path)); rows[0]['answer'] += ' changed'
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        sft_preflight.run_preflight(source, recipe, token, root, resume=True)
    assert len(runner.jobs) == jobs


@pytest.mark.parametrize('count', [1, 2])
def test_completed_unchanged_replay_probe_is_reused_without_new_workers(tmp_path, monkeypatch, count):
    root, result, _, recipe, token, source, runner = replay_probe(tmp_path, monkeypatch, count)
    jobs = len(runner.jobs)
    assert sft_preflight.run_preflight(source, recipe, token, root, resume=True) == result
    assert len(runner.jobs) == jobs


def test_equal_but_incorrect_mixed_finite_states_are_not_a_valid_resume(tmp_path, monkeypatch):
    root, _, _, _, _, _, _ = replay_probe(tmp_path, monkeypatch)
    for name in ['continuous', 'resumed']:
        path = root / name / 'latest.pt'
        saved = torch.load(path, weights_only=True)
        saved['data_state']['epoch'] += 1
        torch.save(saved, path)
    actual = torch.load(root / 'continuous/latest.pt', weights_only=True)
    with pytest.raises(ValueError, match='finite data selection'):
        sft_preflight.verify_updated_probe(root, root / 'zero-init.pt', actual['input_hashes'],
                                           actual['config'], actual['code'])
