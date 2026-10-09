from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from xqgeneral import clean_sft_matches as worker, evaluate_games, inference
from xqgeneral.evidence import atomic_json, code_identity, digest, load_jsonl, manifest
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.rules import legal_moves


@pytest.fixture
def setup(tmp_path, monkeypatch):
    evaluation = tmp_path / 'evaluation.json'
    recipe, token = tmp_path / 'recipe.json', tmp_path / 'token.json'
    for path in [recipe, token]:atomic_json(path, {})
    atomic_json(evaluation, {'sft_recipe': str(recipe), 'token_preflight': str(token), 'validation_examples': 4})
    suite = tmp_path / 'suite';suite.mkdir()
    opening_proof = {'config': {'split': 'validation'}, 'inputs': {}, 'outputs': {}}
    atomic_json(suite / 'manifest.json', opening_proof)
    pipeline, validation = tmp_path / 'pipeline', tmp_path / 'validation'
    source = pipeline / 'sft/adapter.pt';source.parent.mkdir(parents=True);source.write_bytes(b'controlled CPU model fixture')
    identity = {'sha256': digest(source), 'bytes': source.stat().st_size}
    parent = {'outputs': {str(source): identity}, 'verification': {'selected_sft_step': 123}}
    atomic_json(pipeline / 'manifest.json', parent)
    verification = {'actual_clean_initial_sft_completed_before_model_or_reviewer_load': True,
        'split': 'validation', 'all_validation_examples': 4, 'independent_test_used_for_training_or_selection': False,
        'raw_answers_repaired': False, 'raw_validation_by_memory': {'normal': {}, 'zero': {}, 'shuffled': {}}}
    capability = manifest('clean_initial_sft_capability_validation',
        {'pipeline': str(pipeline), 'config': str(evaluation), 'config_sha256': digest(evaluation)}, [source], (), verification)
    atomic_json(validation / 'manifest.json', capability)
    assets = {}
    for name in ['executable', 'weights', 'expert_weights']:
        path = tmp_path / name;path.write_bytes(name.encode());assets[name] = str(path)
    config = {**assets, 'evaluation_config': str(evaluation), 'opening_suite': str(suite), 'opening_count': 2,
              'nodes': [100, 1000, 10000], 'max_plies': 4, 'action_mode': 'explanation', 'move_beams': 4,
              'cuda_visible_devices': '2'}
    config_path = tmp_path / 'config.json';atomic_json(config_path, config)
    monkeypatch.setattr(worker, 'validation_spec', lambda _: ([], {'inputs': {}, 'outputs': {}}, {}))
    monkeypatch.setattr(worker, 'load_openings', lambda _: (evaluate_games.OPENINGS, deepcopy(opening_proof)))
    monkeypatch.setattr(evaluate_games, 'load_openings', lambda _: (evaluate_games.OPENINGS, deepcopy(opening_proof)))
    checked_models = []
    def checked(*args):checked_models.append(args);return source, parent
    monkeypatch.setattr(worker, 'checked_sft', checked)
    return {'pipeline': pipeline, 'validation': validation, 'config': config, 'config_path': config_path,
            'root': tmp_path / 'baseline', 'capability': capability, 'checked_models': checked_models,
            'source': source, 'parent': parent, 'opening_proof': opening_proof}


def run(s, **kwargs):
    return worker.run_baseline(s['pipeline'], s['validation'], 'owned-producer', s['config_path'], s['root'], **kwargs)


@pytest.mark.parametrize('key,value', [
    ('nodes', []), ('nodes', [100, 100]), ('nodes', [True]), ('nodes', [0]), ('nodes', '100'),
    ('opening_count', False), ('max_plies', 0), ('move_beams', 0), ('action_mode', 'legal_move'),
    ('cuda_visible_devices', '0,2'), ('cuda_visible_devices', ''),
])
def test_invalid_budgets_or_constrained_mode_never_admit_a_student(setup, monkeypatch, key, value):
    s = setup;s['config'][key] = value;atomic_json(s['config_path'], s['config'])
    monkeypatch.setattr(worker, 'execute_matches', lambda *a: pytest.fail('player execution before admission'))
    with pytest.raises(ValueError, match='Raw explanation'):run(s)
    assert not s['root'].exists() and not s['checked_models']


@pytest.mark.parametrize('change', ['test_split', 'missing_opening'])
def test_only_the_complete_declared_validation_openings_are_admitted(setup, change):
    s = setup
    if change == 'test_split':s['opening_proof']['config']['split'] = 'test'
    else:s['config']['opening_count'] = 3;atomic_json(s['config_path'], s['config'])
    with pytest.raises(ValueError, match='protected validation'):run(s)
    assert not s['root'].exists() and not s['checked_models']


@pytest.mark.parametrize('change', ['kind', 'pipeline', 'config', 'config_hash', 'unfinished_sft', 'test_split',
                                     'partial_validation', 'test_used', 'repaired', 'missing_ablation'])
def test_partial_or_mismatched_sft_capability_cannot_start_full_matches(setup, monkeypatch, change):
    s = setup;p = deepcopy(s['capability']);v = p['verification']
    if change == 'kind':p['kind'] = 'pilot'
    elif change == 'pipeline':p['config']['pipeline'] = 'other'
    elif change == 'config':p['config']['config'] = 'other'
    elif change == 'config_hash':p['config']['config_sha256'] = '0' * 64
    elif change == 'unfinished_sft':v['actual_clean_initial_sft_completed_before_model_or_reviewer_load'] = False
    elif change == 'test_split':v['split'] = 'test'
    elif change == 'partial_validation':v['all_validation_examples'] -= 1
    elif change == 'test_used':v['independent_test_used_for_training_or_selection'] = True
    elif change == 'repaired':v['raw_answers_repaired'] = True
    elif change == 'missing_ablation':del v['raw_validation_by_memory']['shuffled']
    atomic_json(s['validation'] / 'manifest.json', p)
    monkeypatch.setattr(worker, 'execute_matches', lambda *a: pytest.fail('player execution before admission'))
    with pytest.raises(ValueError):run(s)
    assert not s['checked_models'] and not (s['root'] / 'manifest.json').exists()
    assert json.loads((s['root'] / 'state.json').read_text())['baseline_completion_proven'] is False


def test_selected_model_must_be_the_model_in_the_completed_paired_validation(setup, monkeypatch):
    s = setup;s['parent']['outputs'][str(s['source'])]['sha256'] = '0' * 64
    monkeypatch.setattr(worker, 'execute_matches', lambda *a: pytest.fail('unverified model loaded'))
    with pytest.raises(ValueError, match='selected model actually evaluated'):run(s)
    assert not (s['root'] / 'matches').exists()


class Student:
    calls = 0
    def __init__(self, *args):pass
    def generate(self, record, **kwargs):
        Student.calls += 1
        return json.dumps({'move': legal_moves(record['fen'])[0]})


class Engine:
    calls = 0
    def __init__(self, *args, **kwargs):pass
    def choose_move(self, fen, nodes, *args):
        Engine.calls += 1
        return {'best_move': legal_moves(fen)[0], 'requested_nodes': nodes}
    def close(self):pass


def execute_cpu(monkeypatch):
    Student.calls = Engine.calls = 0
    monkeypatch.setattr(inference, 'Predictor', Student)
    monkeypatch.setattr(evaluate_games, 'Pikafish', Engine)
    def execute(command, log, device):
        assert device == '2'
        assert command[:4] == [sys.executable, '-u', '-m', 'xqgeneral.evaluate_games']
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', command[3:])
            evaluate_games.main()
    monkeypatch.setattr(worker, 'execute_matches', execute)


def test_full_matrix_uses_expanded_node_arguments_and_completed_resume_has_no_inference(setup, monkeypatch):
    s = setup;execute_cpu(monkeypatch)
    result = run(s)
    assert result['games'] == 12 and result['node_budgets'] == [100, 1000, 10000]
    assert result['model_turns'] == Student.calls == 24 and Engine.calls == 24
    assert result['full_explanation_contract_valid_turns'] == 0
    assert result['raw_match_metrics']['censored'] == 12 and result['raw_match_metrics']['draws'] == 0
    assert result['raw_model_answers_repaired'] is result['independent_final_test_used'] is False
    assert result['raw_match_metrics']['rule_legal_constraints'] is False
    command = json.loads((s['root'] / 'commands.jsonl').read_text())
    i = command.index('--nodes');assert command[i+1:i+4] == ['100', '1000', '10000']
    snapshots = {str(p): p.read_bytes() for p in s['root'].rglob('*.json')}
    monkeypatch.setattr(worker, 'execute_matches', lambda *a: pytest.fail('completed baseline executed again'))
    monkeypatch.setattr(worker, 'checked_sft', lambda *a: pytest.fail('completed baseline loaded a large model'))
    assert run(s, resume=True) == result
    assert snapshots == {str(p): p.read_bytes() for p in s['root'].rglob('*.json')}


def test_baseline_resume_reuses_partial_match_checkpoint_and_never_repeats_player_queries(setup, monkeypatch):
    s = setup;execute_cpu(monkeypatch);original = evaluate_games.save_game_checkpoint
    def stop(path, spec, game):
        original(path, spec, game)
        if Path(path).name == 'game-002.json' and len(game['turns']) == 1:raise RuntimeError('controlled interruption')
    with monkeypatch.context() as m:
        m.setattr(evaluate_games, 'save_game_checkpoint', stop)
        with pytest.raises(RuntimeError, match='controlled interruption'):run(s)
    first = s['root'] / 'matches/game-001.json';preserved = first.read_bytes()
    assert not (s['root'] / 'manifest.json').exists()
    result = run(s, resume=True)
    assert result['raw_match_metrics']['saved_turns_reused'] == 5
    assert result['raw_match_metrics']['completed_games_reused'] == 1
    assert Student.calls == Engine.calls == 24 and first.read_bytes() == preserved
    commands = [json.loads(l) for l in (s['root'] / 'commands.jsonl').read_text().splitlines()]
    assert '--resume' not in commands[0] and commands[1][-1] == '--resume'


@pytest.mark.parametrize('change', ['changed_config', 'changed_source'])
def test_resume_rejects_configuration_or_source_changes_without_starting_players(setup, monkeypatch, change):
    s = setup;execute_cpu(monkeypatch);run(s)
    monkeypatch.setattr(worker, 'execute_matches', lambda *a: pytest.fail('changed contract resumed'))
    if change == 'changed_config':
        s['config']['max_plies'] += 1;atomic_json(s['config_path'], s['config'])
    else:
        old = code_identity();monkeypatch.setattr(worker, 'code_identity', lambda: {**old, 'revision': 'other'})
    with pytest.raises(ValueError, match='original baseline contract'):run(s, resume=True)


@pytest.mark.parametrize('change', ['missing_game', 'wrong_budget', 'unbound_checkpoint', 'wrong_budget_metrics'])
def test_readback_requires_every_raw_native_game_and_correct_budget_metrics(setup, monkeypatch, change):
    s = setup;execute_cpu(monkeypatch);run(s)
    dest = s['root'] / 'matches';proof = json.loads((dest / 'manifest.json').read_text())
    games = load_jsonl(dest / 'games.jsonl')
    if change == 'missing_game':
        games.pop();(dest / 'games.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in games))
        proof['outputs'][str(dest / 'games.jsonl')] = {'sha256': digest(dest / 'games.jsonl'), 'bytes': (dest / 'games.jsonl').stat().st_size}
    elif change == 'wrong_budget':
        games[0]['opponent_nodes'] = 999
        (dest / 'games.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in games))
        proof['outputs'][str(dest / 'games.jsonl')] = {'sha256': digest(dest / 'games.jsonl'), 'bytes': (dest / 'games.jsonl').stat().st_size}
    elif change == 'unbound_checkpoint':del proof['outputs'][str(dest / 'progress/game-001.json')]
    else:proof['verification']['by_opponent_nodes']['100']['wins'] = 999
    atomic_json(dest / 'manifest.json', proof)
    with pytest.raises(ValueError):worker.checked_matches(dest, proof['config'], evaluate_games.OPENINGS, BoundInputs(), proof['code'])
