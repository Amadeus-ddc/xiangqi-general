from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from xqgeneral import evaluate_games, inference
from xqgeneral.evidence import atomic_json, load_jsonl
from xqgeneral.evaluate_games import load_game_checkpoint, play_game, restore_game, save_game_checkpoint
from xqgeneral.rules import START_FEN, legal_moves, replay


class Student:
    def __init__(self, *args):
        self.calls = []

    def generate(self, record, **kwargs):
        assert record['history'] == replay(record['initial_fen'], record['moves'])
        self.calls.append(tuple(record['moves']))
        return json.dumps({'move': legal_moves(record['fen'])[0]})

    def generate_moves(self, records, *, beams):
        assert beams == 4 and len(records) == 1
        self.calls.append(tuple(records[0]['moves']))
        return [legal_moves(records[0]['fen'])[0]]


class Engine:
    def __init__(self, *args, **kwargs):
        self.calls, self.closed = [], False

    def choose_move(self, fen, nodes, initial_fen, moves):
        assert replay(initial_fen, moves)[-1] == fen
        self.calls.append(tuple(moves))
        return {'best_move': legal_moves(fen)[0], 'requested_nodes': nodes}

    def close(self):
        self.closed = True


@pytest.mark.parametrize('color', ['red', 'black'])
@pytest.mark.parametrize('mode', ['explanation', 'legal_move'])
@pytest.mark.parametrize('stop', [1, 3, 5])
def test_midgame_resume_preserves_every_raw_turn_and_only_calls_players_for_remaining_moves(tmp_path, color, mode, stop):
    opening = ('supplied', ['b0c2', 'b9c7'])
    full_student, full_engine = Student(), Engine()
    expected = play_game(full_student, full_engine, opening, color, 100, 6, mode)
    assert len(expected['turns']) == 6
    student, engine = Student(), Engine()
    path = tmp_path / 'progress.json'
    spec = evaluate_games._checkpoint_spec(opening, color, 100, 6, mode, 4)

    def checkpoint(game):
        save_game_checkpoint(path, spec, game)
        if len(game['turns']) == stop:
            raise RuntimeError('controlled interruption after a saved turn')

    with pytest.raises(RuntimeError, match='controlled interruption'):
        play_game(student, engine, opening, color, 100, 6, mode, on_progress=checkpoint)
    saved_bytes = path.read_bytes()
    saved = load_game_checkpoint(path, opening, color, 100, 6, mode, 4)
    assert saved['status'] == 'playing' and len(saved['turns']) == stop
    next_student, next_engine = Student(), Engine()
    actual = play_game(next_student, next_engine, opening, color, 100, 6, mode, resume_state=saved)
    assert actual == expected and path.read_bytes() == saved_bytes
    assert student.calls + next_student.calls == full_student.calls
    assert engine.calls + next_engine.calls == full_engine.calls


@pytest.mark.parametrize('reason', ['forfeit', 'censored', 'native_terminal'])
def test_finished_games_replay_without_invoking_either_player(tmp_path, reason):
    class Invalid(Student):
        def generate(self, *args, **kwargs):return '{"move":"a0a0"}'
    class Repeating(Student):
        def generate(self, *args, **kwargs):return '{"move":"c7b9"}'
    if reason == 'forfeit':
        opening, color, limit, student = ('initial', []), 'red', 8, Invalid()
    elif reason == 'censored':
        opening, color, limit, student = ('initial', []), 'red', 2, Student()
    else:
        opening = ('repetition', (['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2)[:-1])
        color, limit, student = 'black', 8, Repeating()
    game = play_game(student, Engine(), opening, color, 100, limit)
    assert game['status'] != 'playing'
    if reason == 'forfeit':assert game['reason'] == 'raw_model_invalid_move_forfeit'
    if reason == 'native_terminal':assert game['reason'] == 'AXF_repetition_or_move_limit'
    path = tmp_path / 'finished.json'
    save_game_checkpoint(path, evaluate_games._checkpoint_spec(opening, color, 100, limit, 'explanation', 4), game)
    saved = load_game_checkpoint(path, opening, color, 100, limit, 'explanation', 4)
    assert play_game(None, None, opening, color, 100, limit, resume_state=saved) == game


@pytest.mark.parametrize('change', ['raw', 'move', 'verification', 'mover', 'ply', 'oracle_used',
                                  'moves', 'status', 'winner', 'initial_fen', 'book', 'nodes', 'engine_nodes'])
def test_native_replay_rejects_altered_game_even_with_a_recomputed_checkpoint_hash(tmp_path, change):
    opening = ('initial', [])
    game = play_game(Student(), Engine(), opening, 'red', 100, 4)
    bad = deepcopy(game)
    first, opponent = bad['turns'][0], bad['turns'][1]
    if change == 'raw':first['raw'] = '{"move":"a0a0"}'
    if change == 'move':first['move'] = 'a0a0'
    if change == 'verification':first['verification']['valid'] = 0
    if change == 'mover':first['mover'] = 'black'
    if change == 'ply':first['ply'] = True
    if change == 'oracle_used':first['model_oracle_used'] = True
    if change == 'moves':bad['moves'][0] = 'a0a0'
    if change == 'status':bad['status'] = 'completed'
    if change == 'winner':bad['winner'] = 'red'
    if change == 'initial_fen':bad['initial_fen'] = START_FEN.replace(' 0 1', ' 0 2')
    if change == 'book':bad['opening_moves'] = ['h0g2']
    if change == 'nodes':bad['opponent_nodes'] = 1000
    if change == 'engine_nodes':opponent['oracle']['requested_nodes'] = 1000
    path = tmp_path / 'corrupted.json'
    save_game_checkpoint(path, evaluate_games._checkpoint_spec(opening, 'red', 100, 4, 'explanation', 4), bad)
    with pytest.raises(ValueError, match='Saved'):
        load_game_checkpoint(path, opening, 'red', 100, 4, 'explanation', 4)


def test_resume_rejects_hash_changes_budget_changes_and_moves_after_native_terminal(tmp_path):
    opening = ('initial', [])
    game = play_game(Student(), Engine(), opening, 'red', 100, 2)
    path = tmp_path / 'progress.json'
    spec = evaluate_games._checkpoint_spec(opening, 'red', 100, 2, 'explanation', 4)
    save_game_checkpoint(path, spec, game)
    data = json.loads(path.read_text());data['game']['moves'][0] = 'a0a0';atomic_json(path, data)
    with pytest.raises(ValueError, match='hash'):
        load_game_checkpoint(path, opening, 'red', 100, 2, 'explanation', 4)
    save_game_checkpoint(path, spec, game)
    with pytest.raises(ValueError, match='specification'):
        load_game_checkpoint(path, opening, 'red', 100, 3, 'explanation', 4)
    bad = deepcopy(game);bad['turns'].append(deepcopy(game['turns'][0]))
    with pytest.raises(ValueError, match='terminal position or its ply limit'):
        restore_game(opening, 'red', 100, 2, bad)


def cli_arguments(tmp_path, *, limit=4):
    artifacts = {}
    for name in ['checkpoint', 'engine', 'nnue', 'expert']:
        p = tmp_path / name;p.write_text(name);artifacts[name] = p
    dest = tmp_path / 'matches'
    argv = ['evaluate_games', '--checkpoint', str(artifacts['checkpoint']),
            '--executable', str(artifacts['engine']), '--weights', str(artifacts['nnue']),
            '--expert-weights', str(artifacts['expert']), '--nodes', '100', '1000',
            '--max-plies', str(limit), '--output', str(dest)]
    return argv, dest, artifacts


def test_cli_resume_reuses_completed_games_and_active_turns_then_completed_replay_loads_no_models(tmp_path, monkeypatch):
    argv, dest, artifacts = cli_arguments(tmp_path)
    students, engines = [], []
    def student(*args):
        value = Student();students.append(value);return value
    def engine(*args, **kwargs):
        value = Engine();engines.append(value);return value
    monkeypatch.setattr(inference, 'Predictor', student)
    monkeypatch.setattr(evaluate_games, 'Pikafish', engine)
    original = save_game_checkpoint
    def interrupted(path, spec, game):
        original(path, spec, game)
        if Path(path).name == 'game-002.json' and len(game['turns']) == 1:
            raise RuntimeError('controlled CLI interruption')
    monkeypatch.setattr(evaluate_games, 'save_game_checkpoint', interrupted)
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(RuntimeError, match='CLI interruption'):evaluate_games.main()
    assert not (dest / 'manifest.json').exists() and all(e.closed for e in engines)
    first_game, contract = (dest / 'game-001.json').read_bytes(), (dest / 'contract.json').read_bytes()
    monkeypatch.setattr(evaluate_games, 'save_game_checkpoint', original)
    monkeypatch.setattr(sys, 'argv', [*argv, '--resume'])
    evaluate_games.main()
    games = load_jsonl(dest / 'games.jsonl');metrics = json.loads((dest / 'metrics.json').read_text())
    assert len(games) == 8 and all(len(g['turns']) == 4 for g in games)
    assert metrics['completed_games_reused'] == 1 and metrics['saved_turns_reused'] == 5
    assert metrics['censored'] == 8 and metrics['draws'] == 0 and metrics['model_oracle_repairs'] == 0
    assert sum(len(s.calls) for s in students) == sum(len(e.calls) for e in engines) == 16
    assert (dest / 'game-001.json').read_bytes() == first_game and (dest / 'contract.json').read_bytes() == contract
    proof = json.loads((dest / 'manifest.json').read_text())
    assert len(proof['outputs']) == 19 and all(e.closed for e in engines)
    preserved = {p: p.read_bytes() for p in dest.rglob('*.json')}
    def forbidden(*args, **kwargs):raise AssertionError('Completed replay must not load players')
    monkeypatch.setattr(inference, 'Predictor', forbidden);monkeypatch.setattr(evaluate_games, 'Pikafish', forbidden)
    evaluate_games.main()
    assert all(p.read_bytes() == data for p, data in preserved.items())


@pytest.mark.parametrize('change', ['checkpoint', 'engine', 'nnue', 'expert', 'nodes', 'plies', 'mode', 'beams', 'source'])
def test_cli_resume_rejects_changed_inputs_or_recipe_before_any_player_loads(tmp_path, monkeypatch, change):
    argv, dest, artifacts = cli_arguments(tmp_path)
    monkeypatch.setattr(inference, 'Predictor', Student);monkeypatch.setattr(evaluate_games, 'Pikafish', Engine)
    monkeypatch.setattr(sys, 'argv', argv);evaluate_games.main()
    if change in artifacts:artifacts[change].write_text('changed')
    if change == 'nodes':argv[argv.index('--nodes') + 1] = '101'
    if change == 'plies':argv[argv.index('--max-plies') + 1] = '5'
    if change == 'mode':argv += ['--action-mode', 'legal_move']
    if change == 'beams':argv += ['--move-beams', '2']
    if change == 'source':
        original = evaluate_games.code_identity
        monkeypatch.setattr(evaluate_games, 'code_identity', lambda: dict(original(), revision='changed'))
    def forbidden(*args, **kwargs):raise AssertionError('Changed resume must fail before loading players')
    monkeypatch.setattr(inference, 'Predictor', forbidden);monkeypatch.setattr(evaluate_games, 'Pikafish', forbidden)
    monkeypatch.setattr(sys, 'argv', [*argv, '--resume'])
    with pytest.raises(ValueError, match='original input, source and evaluation contract'):evaluate_games.main()


@pytest.mark.parametrize('change', ['gap', 'unbound', 'unexpected', 'published', 'missing_output_hash'])
def test_cli_resume_rejects_broken_checkpoint_coverage_and_completed_output_integrity(tmp_path, monkeypatch, change):
    argv, dest, artifacts = cli_arguments(tmp_path)
    monkeypatch.setattr(inference, 'Predictor', Student);monkeypatch.setattr(evaluate_games, 'Pikafish', Engine)
    monkeypatch.setattr(sys, 'argv', argv);evaluate_games.main()
    if change == 'gap':
        for p in [dest / 'game-001.json', dest / 'progress/game-001.json']:p.rename(p.with_suffix('.preserved'))
    if change == 'unbound':
        p = dest / 'progress/game-001.json';p.rename(p.with_suffix('.preserved'))
    if change == 'unexpected':
        (dest / 'progress/game-999.json').write_bytes((dest / 'progress/game-001.json').read_bytes())
    if change == 'published':
        p = dest / 'game-001.json';d = json.loads(p.read_text());d['winner'] = 'red';atomic_json(p, d)
    if change == 'missing_output_hash':
        p = dest / 'manifest.json';d = json.loads(p.read_text());d['outputs'].pop(str(dest / 'progress/game-001.json'));atomic_json(p, d)
    def forbidden(*args, **kwargs):raise AssertionError('Broken progress cannot load players')
    monkeypatch.setattr(inference, 'Predictor', forbidden);monkeypatch.setattr(evaluate_games, 'Pikafish', forbidden)
    monkeypatch.setattr(sys, 'argv', [*argv, '--resume'])
    with pytest.raises(ValueError):evaluate_games.main()


def test_resume_requires_an_existing_bound_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['evaluate_games', '--checkpoint', 'missing',
                                   '--output', str(tmp_path / 'missing'), '--resume'])
    with pytest.raises(FileNotFoundError, match='existing match'):evaluate_games.main()
