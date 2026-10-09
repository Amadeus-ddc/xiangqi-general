from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, write_jsonl
from xqgeneral.human_games import assigned_split
from xqgeneral.match_openings import checked_opening, load_openings, prepare_openings
from xqgeneral.rules import START_FEN, legal_moves, replay


def source_rows(split='validation'):
    ids = []
    for i in range(10000):
        name = 'recorded-' + f'{i:064x}'
        if assigned_split(name, 20261051) == split:
            ids.append(name)
        if len(ids) == 3:break
    result = []
    for i, book in enumerate([['b0c2', 'b9c7'], ['h0g2', 'h9g7'], ['b2e2', 'h9g7']]):
        history = replay(START_FEN, book)
        result.append({'id': f'original-{i}', 'game_id': ids[i], 'recorded_source_game_id': ids[i],
            'split': split, 'stage': 'static_current', 'task_type': 'count',
            'initial_fen': START_FEN, 'moves': book, 'history': history, 'fen': history[-1],
            'feature_key': history_key(history), 'recorded_source_kind': 'recorded_human_match',
            'answer': 'This answer must never become a match label'})
    return result


def make_source(root, rows=None, split='validation'):
    root.mkdir()
    rows = source_rows(split) if rows is None else rows
    path = root / f'{split}.jsonl';write_jsonl(path, rows)
    # Unrelated training and test files are never read by suite preparation.
    (root / 'train.jsonl').write_text('not valid JSON; never read training answers\n')
    proof = {'status': 'complete', 'kind': 'isolated_recorded_engine_four_course_questions',
             'config': {'split_seed': 20261051},
             'outputs': {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size}}}
    atomic_json(root / 'manifest.json', proof)
    return root


def pack(tmp_path, split='validation'):
    data, out = make_source(tmp_path / 'data', split=split), tmp_path / 'suite'
    prepare_openings(data, out, split=split, count=2, plies=2)
    return data, out


def rebind_output(out, rows):
    write_jsonl(out / 'openings.jsonl', rows)
    proof = json.loads((out / 'manifest.json').read_text())
    path = out / 'openings.jsonl'
    proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(out / 'manifest.json', proof)


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_completed_suite_reads_only_requested_protected_source_and_contains_no_labels(tmp_path, split):
    data, out = pack(tmp_path, split)
    rows, proof = load_openings(out)
    assert len(rows) == 2 and proof['verification']['source_rows_read'] == 3
    assert {r['split'] for r in rows} == {split}
    assert len({r['game_id'] for r in rows}) == len({r['fen'] for r in rows}) == 2
    assert all('answer' not in r and 'question' not in r and 'future_moves' not in r for r in rows)
    assert not proof['verification']['student_or_teacher_model_loaded']
    assert not proof['verification']['answers_or_recorded_future_used']
    assert set(proof['inputs']) == {str(data / 'manifest.json'), str(data / f'{split}.jsonl')}
    assert not proof['verification']['opening_geometry_unseen_during_training_proven']


def test_selection_is_independent_of_file_order_answers_and_repeated_question_rows(tmp_path):
    rows = source_rows(); duplicate = dict(rows[0], id='zzz-duplicate', answer='different label')
    mirrored = dict(rows[1], id='derived', augmentation_parent=rows[1]['id'], answer='unused')
    a = make_source(tmp_path / 'a', rows + [duplicate, mirrored])
    b = make_source(tmp_path / 'b', [mirrored, duplicate, *reversed(rows)])
    oa, ob = tmp_path / 'oa', tmp_path / 'ob'
    prepare_openings(a, oa, count=3, plies=2);prepare_openings(b, ob, count=3, plies=2)
    assert load_jsonl(oa / 'openings.jsonl') == load_jsonl(ob / 'openings.jsonl')
    assert load_openings(oa)[1]['verification']['eligible_original_recorded_games'] == 3


@pytest.mark.parametrize('change', ['split', 'game_id', 'history', 'feature_key', 'source_kind', 'prefix_conflict'])
def test_rebound_source_metadata_cannot_silently_change_original_ownership_or_history(tmp_path, change):
    rows = source_rows()
    if change == 'split':rows[-1]['split'] = 'train'
    if change == 'game_id':rows[0]['recorded_source_game_id'] = rows[1]['game_id']
    if change == 'history':rows[0]['history'][-1] = START_FEN
    if change == 'feature_key':rows[0]['feature_key'] = 'wrong'
    if change == 'source_kind':rows += [dict(rows[0], id='conflict', recorded_source_kind='published_recorded_match')]
    if change == 'prefix_conflict':
        rows += [dict(rows[1], id='conflict', game_id=rows[0]['game_id'],
                      recorded_source_game_id=rows[0]['game_id'])]
    data = make_source(tmp_path / 'data', rows)
    with pytest.raises(ValueError):prepare_openings(data, tmp_path / 'suite', count=2, plies=2)
    assert (tmp_path / 'suite/failure.json').exists()
    with pytest.raises(FileExistsError):prepare_openings(data, tmp_path / 'suite', count=2, plies=2)


@pytest.mark.parametrize('problem', ['split', 'count_bool', 'count_zero', 'plies_bool', 'plies_zero', 'seed_bool'])
def test_invalid_budgets_are_rejected_before_source_or_output_access(tmp_path, problem):
    kwargs = {'count': 2, 'plies': 2}
    kwargs.update({'split': 'train'} if problem == 'split' else {
        'count_bool': {'count': True}, 'count_zero': {'count': 0}, 'plies_bool': {'plies': True},
        'plies_zero': {'plies': 0}, 'seed_bool': {'seed': True}}[problem])
    with pytest.raises(ValueError):prepare_openings(tmp_path / 'missing', tmp_path / 'suite', **kwargs)
    assert not (tmp_path / 'suite').exists()


def test_duplicate_geometric_openings_do_not_fill_requested_budget(tmp_path):
    rows = source_rows();rows[1].update({k: deepcopy(rows[0][k]) for k in
                                      ['initial_fen', 'moves', 'history', 'fen', 'feature_key']})
    data = make_source(tmp_path / 'data', rows)
    with pytest.raises(ValueError, match='Insufficient'):
        prepare_openings(data, tmp_path / 'suite', count=3, plies=2)
    assert not (tmp_path / 'suite/manifest.json').exists()


def test_terminal_prefix_is_not_admitted_as_an_opening(tmp_path):
    rows = source_rows()
    past = ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2
    history = replay(START_FEN, past)
    rows[0].update(moves=past, history=history, fen=history[-1], feature_key=history_key(history))
    data = make_source(tmp_path / 'data', rows)
    with pytest.raises(ValueError, match='Insufficient'):
        prepare_openings(data, tmp_path / 'suite', count=1, plies=8)


@pytest.mark.parametrize('change', ['native_history', 'source_prefix', 'source_provenance', 'duplicate', 'split', 'source_hash'])
def test_loader_checks_native_history_and_exact_source_even_if_output_hashes_are_rebound(tmp_path, change):
    data, out = pack(tmp_path);rows = load_jsonl(out / 'openings.jsonl')
    if change == 'native_history':rows[0]['history'][-1] = START_FEN
    if change == 'source_prefix':
        past = ['a0a1', 'a9a8'];history = replay(START_FEN, past)
        rows[0].update(moves=past, history=history, fen=history[-1], feature_key=history_key(history))
    if change == 'source_provenance':rows[0]['source_row_id'] = 'nonexistent'
    if change == 'duplicate':rows[1] = dict(rows[0], id='different-id')
    if change == 'split':rows[0]['split'] = 'test'
    if change == 'source_hash':
        with (data / 'validation.jsonl').open('a') as f:f.write('\n')
    rebind_output(out, rows)
    with pytest.raises(ValueError):load_openings(out)


def test_loader_refuses_unfinished_suite(tmp_path):
    data, out = pack(tmp_path)
    proof = json.loads((out / 'manifest.json').read_text());proof['status'] = 'failed'
    atomic_json(out / 'manifest.json', proof)
    with pytest.raises(ValueError, match='completed'):load_openings(out)


def test_match_cli_runs_each_recorded_opening_as_both_colors_at_each_budget(tmp_path, monkeypatch):
    from xqgeneral import evaluate_games, inference
    data, out = pack(tmp_path);expected = load_openings(out)[0]
    inputs = {}
    for name in ['checkpoint', 'engine', 'weights', 'expert']:
        inputs[name] = tmp_path / name;inputs[name].write_text(name)
    class Student:
        def __init__(self, checkpoint, expert):
            assert checkpoint == str(inputs['checkpoint']) and expert == str(inputs['expert'])
        def generate(self, record, **kwargs):
            assert record['history'] == replay(record['initial_fen'], record['moves'])
            return json.dumps({'move': legal_moves(record['fen'])[0]})
    class Engine:
        def __init__(self, *args, **kwargs):pass
        def choose_move(self, fen, nodes, initial_fen, moves):
            assert replay(initial_fen, moves)[-1] == fen
            return {'best_move': legal_moves(fen)[0], 'requested_nodes': nodes}
        def close(self):pass
    monkeypatch.setattr(inference, 'Predictor', Student);monkeypatch.setattr(evaluate_games, 'Pikafish', Engine)
    dest = tmp_path / 'matches'
    monkeypatch.setattr(sys, 'argv', ['evaluate_games', '--checkpoint', str(inputs['checkpoint']),
        '--executable', str(inputs['engine']), '--weights', str(inputs['weights']), '--opening-suite', str(out),
        '--expert-weights', str(inputs['expert']), '--nodes', '100', '1000', '--max-plies', '2', '--output', str(dest)])
    evaluate_games.main()
    games = load_jsonl(dest / 'games.jsonl');proof = json.loads((dest / 'manifest.json').read_text())
    assert len(games) == 8
    assert {(g['opening'], g['model_color'], g['opponent_nodes']) for g in games} == {
        (r['id'], c, n) for r in expected for c in ['red', 'black'] for n in [100, 1000]}
    assert all(g['opening_split'] == 'validation' and g['opening_moves'] in [r['moves'] for r in expected]
               and g['initial_fen'] == START_FEN for g in games)
    assert proof['verification']['both_model_colors_at_each_opening_and_budget']
    assert proof['verification']['model_oracle_repairs'] == 0
    assert str(out / 'manifest.json') in proof['inputs']
    assert str(data / 'validation.jsonl') in proof['inputs'] and str(inputs['expert']) in proof['inputs']


@pytest.mark.parametrize('mutation', ['checkpoint', 'suite'])
def test_inputs_changed_during_matches_never_receive_a_completed_manifest(tmp_path, monkeypatch, mutation):
    from xqgeneral import evaluate_games, inference
    data, out = pack(tmp_path)
    model = tmp_path / 'checkpoint';model.write_text('before')
    weights = tmp_path / 'weights';weights.write_text('fixture')
    class Student:
        def __init__(self, *args):pass
        def generate(self, record, **kwargs):
            target = model if mutation == 'checkpoint' else out / 'openings.jsonl'
            target.write_text(target.read_text() + '\n')
            return '{"move":"a0a0"}'
    class Engine:
        def __init__(self, *args, **kwargs):pass
        def choose_move(self, fen, nodes, *args):return {'best_move': legal_moves(fen)[0]}
        def close(self):pass
    monkeypatch.setattr(inference, 'Predictor', Student);monkeypatch.setattr(evaluate_games, 'Pikafish', Engine)
    dest = tmp_path / 'matches'
    monkeypatch.setattr(sys, 'argv', ['evaluate_games', '--checkpoint', str(model),
        '--executable', str(weights), '--weights', str(weights), '--expert-weights', str(weights),
        '--opening-suite', str(out), '--nodes', '100', '--output', str(dest)])
    with pytest.raises(ValueError, match='changed'):evaluate_games.main()
    assert (dest / 'games.jsonl').exists() and not (dest / 'manifest.json').exists()


def test_duplicate_node_budgets_are_rejected_before_loading_a_model(tmp_path, monkeypatch):
    from xqgeneral import evaluate_games
    monkeypatch.setattr(sys, 'argv', ['evaluate_games', '--checkpoint', 'missing',
        '--nodes', '100', '100', '--output', str(tmp_path / 'matches')])
    with pytest.raises(ValueError, match='distinct'):evaluate_games.main()
    assert not (tmp_path / 'matches').exists()


def test_bad_opening_suite_is_rejected_before_model_loading(tmp_path, monkeypatch):
    from xqgeneral import evaluate_games, inference
    data, out = pack(tmp_path)
    (out / 'openings.jsonl').write_text('corrupted\n')
    def forbidden(*args, **kwargs):raise AssertionError('No model may load before suite admission')
    monkeypatch.setattr(inference, 'Predictor', forbidden)
    monkeypatch.setattr(sys, 'argv', ['evaluate_games', '--checkpoint', 'missing',
        '--opening-suite', str(out), '--output', str(tmp_path / 'matches')])
    with pytest.raises(ValueError):evaluate_games.main()
    assert not (tmp_path / 'matches').exists()
