"""Real raw-evaluation input binding with controlled model/training guards."""
import json
from pathlib import Path
import subprocess

import pytest

from test_clean_sft_evaluation import ControlledEvaluation, artifact, setup_evaluation
from test_feature_store import block
from xqgeneral import clean_sft_evaluation as evaluation
from xqgeneral.evidence import atomic_json, digest
from xqgeneral.feature_store import FeatureStoreWriter, feature_paths


def sharded_case(tmp_path, monkeypatch):
    pipeline, path, runner = setup_evaluation(tmp_path, monkeypatch)
    config = json.loads(path.read_text())
    rows, token, recipe = evaluation.validation_spec(config)
    source, parent = evaluation.checked_sft(pipeline, config, recipe)
    # Keep the existing model-guard fixture. Exercise the real sharded cache
    # catalog, raw CLI, BoundInputs and completion checking at this interface.
    writer = FeatureStoreWriter(tmp_path / 'sharded-cache', [0, 3], 'a' * 64, expert_dim=8)
    writer.append(block([r['feature_key'] for r in rows[:8]]))
    writer.append(block([r['feature_key'] for r in rows[8:]], start=4))
    writer.finish()
    cache = writer.output / 'manifest.json'
    config['features'] = str(cache)
    token['cache_header_only_input'] = {'path': str(cache), **artifact(cache)}
    atomic_json(path, config)
    monkeypatch.setattr(evaluation, 'validation_spec', lambda _: (rows, token, recipe))
    monkeypatch.setattr(evaluation, 'checked_sft', lambda *args: (source, parent))
    return pipeline, path, runner, cache


def test_actual_sharded_raw_input_inventory_reaches_all_three_groups(tmp_path, monkeypatch):
    pipeline, config, runner, cache = sharded_case(tmp_path, monkeypatch)
    controlled = ControlledEvaluation(runner, monkeypatch)
    output = tmp_path / 'paired'
    result = evaluation.run_evaluation(pipeline, 'owned-sft', config, output)
    assert result['all_validation_examples'] == 16 and controlled.loads == 3
    for memory in ['normal', 'zero', 'shuffled']:
        proof = json.loads((output / memory / 'manifest.json').read_text())
        assert {str(p) for p in feature_paths(cache)} <= set(proof['inputs'])
    complete = json.loads((output / 'manifest.json').read_text())
    assert {str(p) for p in feature_paths(cache)} <= set(complete['inputs'])


@pytest.mark.parametrize('problem', ['missing_shard', 'foreign_input', 'changed_shard'])
def test_sharded_input_contract_still_rejects_missing_foreign_or_changed_dependencies(tmp_path, monkeypatch, problem):
    pipeline, config, runner, cache = sharded_case(tmp_path, monkeypatch)
    controlled = ControlledEvaluation(runner, monkeypatch)
    paths = feature_paths(cache)
    shard = next(p for p in paths if str(p).endswith('.pt'))

    def corrupt(command, **kwargs):
        result = controlled(command, **kwargs)
        if 'xqgeneral.evaluate_explanations' in command:
            output = Path(command[command.index('--output') + 1])
            proof = json.loads((output / 'manifest.json').read_text())
            if problem == 'missing_shard':
                proof['inputs'].pop(str(shard))
            elif problem == 'foreign_input':
                foreign = tmp_path / 'foreign.bin';foreign.write_bytes(b'not an inference dependency')
                proof['inputs'][str(foreign)] = artifact(foreign)
            else:
                shard.write_bytes(shard.read_bytes() + b'changed')
            atomic_json(output / 'manifest.json', proof)
        return result

    monkeypatch.setattr(subprocess, 'run', corrupt)
    with pytest.raises(ValueError):
        evaluation.run_evaluation(pipeline, 'owned-sft', config, tmp_path / 'rejected')
    assert controlled.loads == 1


def completed_for_reuse(tmp_path, monkeypatch):
    pipeline, config, runner, cache = sharded_case(tmp_path, monkeypatch)
    controlled = ControlledEvaluation(runner, monkeypatch)
    original = tmp_path / 'original'
    with monkeypatch.context() as previous:
        previous.setenv('XQGENERAL_CODE_REVISION', 'controlled-previous-revision')
        evaluation.run_evaluation(pipeline, 'owned-sft', config, original)
    settings = json.loads(config.read_text())
    settings['reuse_raw_validation'] = {
        memory: {'path': str(original / memory), 'manifest_sha256': digest(original / memory / 'manifest.json')}
        for memory in ['normal', 'zero', 'shuffled']}
    path = tmp_path / 'resumed-config.json'
    atomic_json(path, settings)
    return pipeline, path, original, controlled


def test_declared_completed_raw_groups_reuse_all_model_answers_and_engine_judgments(tmp_path, monkeypatch):
    pipeline, config, original, controlled = completed_for_reuse(tmp_path, monkeypatch)
    bindings = {memory: digest(original / memory / 'manifest.json') for memory in ['normal', 'zero', 'shuffled']}
    output = tmp_path / 'resumed'
    result = evaluation.run_evaluation(pipeline, 'owned-sft', config, output)
    assert result['reused_raw_validation_memories'] == ['normal', 'zero', 'shuffled']
    assert controlled.loads == 3
    assert sum('xqgeneral.evaluate_explanations' in c for c in controlled.commands) == 3
    assert not (output / 'normal').exists()
    assert all(digest(original / memory / 'manifest.json') == binding for memory, binding in bindings.items())
    query = next(c for c in controlled.commands if 'xqgeneral.review_explanations' in c and 'prepare' in c and
                 str(output / 'queries') in c)
    assert str(original / 'normal/judged-predictions.jsonl') in query
    assert evaluation.run_evaluation(pipeline, 'owned-sft', config, output, resume=True) == result


def test_only_coordinator_source_can_differ_when_completed_raw_is_reused(tmp_path, monkeypatch):
    pipeline, config, original, controlled = completed_for_reuse(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    for memory in ['normal', 'zero', 'shuffled']:
        path = original / memory / 'manifest.json'
        proof = json.loads(path.read_text())
        proof['code']['source_sha256']['src/xqgeneral/clean_sft_evaluation.py'] = '0' * 64
        atomic_json(path, proof)
        settings['reuse_raw_validation'][memory]['manifest_sha256'] = digest(path)
    atomic_json(config, settings)
    result = evaluation.run_evaluation(pipeline, 'owned-sft', config, tmp_path / 'new-coordinator')
    assert len(result['reused_raw_validation_memories']) == 3 and controlled.loads == 3


@pytest.mark.parametrize('problem', ['changed_modeling', 'changed_training', 'missing_source', 'changed_budget', 'manifest_digest'])
def test_completed_raw_reuse_rejects_changed_inference_source_budget_or_digest(tmp_path, monkeypatch, problem):
    pipeline, config, original, controlled = completed_for_reuse(tmp_path, monkeypatch)
    path = original / 'normal/manifest.json'
    proof = json.loads(path.read_text());settings = json.loads(config.read_text())
    if problem.startswith('changed_') and problem in ['changed_modeling', 'changed_training']:
        filename = problem.removeprefix('changed_')
        proof['code']['source_sha256']['src/xqgeneral/' + filename + '.py'] = '0' * 64
    elif problem == 'missing_source':
        proof['code']['source_sha256'].pop('src/xqgeneral/inference.py')
    elif problem == 'changed_budget':
        proof['config']['max_new_tokens'] += 1
    atomic_json(path, proof)
    if problem != 'manifest_digest':
        settings['reuse_raw_validation']['normal']['manifest_sha256'] = digest(path)
    else:
        settings['reuse_raw_validation']['normal']['manifest_sha256'] = '0' * 64
    atomic_json(config, settings)
    with pytest.raises(ValueError):
        evaluation.run_evaluation(pipeline, 'owned-sft', config, tmp_path / 'rejected-reuse')
    assert controlled.loads == 3


@pytest.mark.parametrize('declaration', [None, [], {'foreign': {}}, {'normal': {'path': 'x', 'manifest_sha256': 'bad'}}])
def test_malformed_reuse_declarations_are_rejected(declaration):
    with pytest.raises(ValueError):
        evaluation.raw_reuse_spec({'reuse_raw_validation': declaration})


def test_completed_raw_manifest_must_exist_before_any_replacement_model_is_loaded(tmp_path, monkeypatch):
    pipeline, config, runner, _ = sharded_case(tmp_path, monkeypatch)
    controlled = ControlledEvaluation(runner, monkeypatch)
    settings = json.loads(config.read_text())
    settings['reuse_raw_validation'] = {'normal': {'path': str(tmp_path / 'unfinished'), 'manifest_sha256': '0' * 64}}
    atomic_json(config, settings)
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.run_evaluation(pipeline, 'owned-sft', config, tmp_path / 'no-replacement')
    assert controlled.loads == 0
