"""Native tactical-import contracts using small, explicitly controlled PGNs."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from xqgeneral.evidence import atomic_json, load_jsonl, manifest, write_jsonl
from xqgeneral.human_games import assigned_split, parse_game
from xqgeneral.recorded_search_inputs import checked_game
from xqgeneral.rules import START_FEN, play, replay
from xqgeneral.tactical_games import import_tactics, native_line, source_inventory


def pgn(fen=START_FEN, moves=('b0c2', 'b9c7'), *, players=True, event='controlled tactical line'):
    headers = {'Game': 'Chinese Chess', 'Event': event, 'FEN': fen, 'Result': '1-0'}
    if players:
        headers.update(Red='受控甲', Black='受控乙')
    return ('\n'.join(f'[{key} {json.dumps(value,ensure_ascii=False)}]' for key,value in headers.items()) +
            '\n\n' + ' '.join(moves) + ' 1-0\n').encode()


def origin(raw, category='midgame'):
    return {'repository': 'controlled source', 'revision': '0' * 40,
            'paths': ['Dataset/中局/controlled.pgn'], 'categories': [category],
            'content_sha256': hashlib.sha256(raw).hexdigest()}


def source_repo(tmp_path, files=None):
    root = tmp_path / 'source'
    root.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    if files is None:
        files = {'Dataset/全盤戰術/full.pgn': pgn(),
                 'Dataset/中局/fragment.pgn': pgn(play(START_FEN, 'b0c2'), ['h9g7'], players=False),
                 'Dataset/殘局/same.pgn': pgn(play(START_FEN, 'b0c2'), ['h9g7'], players=False),
                 'Dataset/殺局_殺法_練習題/invalid.pgn': pgn(moves=['a0a9']),
                 'Dataset/對局/ignored.pgn': pgn(), 'LICENSE': b'Controlled test fixture license, not source rights.'}
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(['git', '-c', 'user.name=Controlled Fixture', '-c', 'user.email=fixture@example.invalid',
                    '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Controlled source'], cwd=root, check=True)
    return root, subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def test_missing_participants_remain_missing_without_weakening_full_match_default():
    raw = pgn(play(START_FEN, 'b0c2'), ['h9g7'], players=False)
    with pytest.raises(ValueError, match='both players'):
        parse_game(raw)
    game, error = native_line((raw, origin(raw)))
    assert error is None and game['declared_participants'] == {'Red': None, 'Black': None}
    assert game['players'] == [] and 'Red' not in game['headers'] and 'Black' not in game['headers']
    assert not game['pre_fragment_game_history_available']
    assert game['history'] == replay(game['initial_fen'], game['moves'])
    assert not game['provided_line_is_best_move_label'] and not game['neural_explanations_generated']
    assert game['source_kind'] == 'recorded_tactical_line'
    game['split'] = assigned_split(game['game_id'], 20261051)
    checked_game(game, 20261051)


def test_initial_geometry_with_later_clock_does_not_invent_pre_fragment_history():
    fen = ' '.join(START_FEN.split()[:4] + ['8', '12'])
    game, error = native_line((pgn(fen=fen), origin(pgn(fen=fen))))
    assert error is None and not game['pre_fragment_game_history_available']
    full, error = native_line((pgn(), origin(pgn())))
    assert error is None and full['pre_fragment_game_history_available']


@pytest.mark.parametrize('value', [None, 1, 'false'])
def test_participant_requirement_cannot_be_implicitly_coerced(value):
    with pytest.raises(ValueError, match='boolean'):
        parse_game(pgn(), require_players=value)


def test_changed_raw_identity_fails_before_native_parse():
    raw = pgn()
    with pytest.raises(ValueError, match='bytes changed'):
        native_line((raw + b' ', origin(raw)))


@pytest.mark.parametrize('raw', [pgn(moves=['a0a9']), pgn().replace(b' 1-0\n', b' 0-1\n')])
def test_illegal_or_mismatched_solution_is_quarantined_without_repair(raw):
    game, rejected = native_line((raw, origin(raw)))
    assert game is None and rejected['content_sha256'] == origin(raw)['content_sha256']
    assert rejected['error']


def test_native_import_preserves_blob_aliases_splits_and_raw_bytes(tmp_path):
    source, revision = source_repo(tmp_path)
    output = tmp_path / 'import'
    proof = import_tactics(source, revision, output, workers=1)
    assert proof['source_category_files'] == 4 and proof['unique_git_blobs_examined'] == 3
    assert proof['new_unique_tactical_lines'] == 2 and proof['quarantined_blobs'] == 1
    assert proof['nonstandard_initial_position_fragments'] == proof['lines_missing_one_or_both_declared_participants'] == 1
    games = load_jsonl(output / 'games.jsonl')
    fragment = next(game for game in games if not game['pre_fragment_game_history_available'])
    assert len(fragment['source']['paths']) == 2 and fragment['source']['categories'] == ['endgame', 'midgame']
    for game in games:
        assert game['split'] == assigned_split(game['game_id'], 20261051)
        assert game['history'] == replay(game['initial_fen'], game['moves'])
        checked_game(game, 20261051)
        raw = (output / 'sources' / (game['source']['blob_sha1'] + '.pgn')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == game['source']['content_sha256']
    assert not proof['training_questions_or_neural_labels_generated'] and not proof['gpu_or_teacher_loaded']
    with pytest.raises(FileExistsError):
        import_tactics(source, revision, output, workers=1)


def test_prior_canonical_lines_are_not_reintroduced_with_new_attribution(tmp_path):
    source, revision = source_repo(tmp_path)
    previous = tmp_path / 'previous'
    previous.mkdir()
    game = parse_game(pgn(event='earlier recorded provenance'))
    game['split'] = assigned_split(game['game_id'], 20261051)
    write_jsonl(previous / 'games.jsonl', [game])
    atomic_json(previous / 'manifest.json', manifest('recorded_games_native_import', {'seed': 20261051},
        [], [previous / 'games.jsonl'], {'all_retained_moves_native_legal': True,
            'full_history_terminal_checks_passed': True, 'game_split_assigned_before_questions': True}))
    output = tmp_path / 'new'
    proof = import_tactics(source, revision, output, previous=[previous], workers=1)
    assert proof['new_unique_tactical_lines'] == 1 and proof['duplicate_canonical_lines'] == 1
    duplicate = load_jsonl(output / 'duplicates.jsonl')[0]
    assert duplicate['game_id'] == game['game_id'] and duplicate['original_split'] == game['split']
    assert duplicate['source']['paths'] == ['Dataset/全盤戰術/full.pgn']


def test_serial_and_two_worker_native_imports_have_identical_ordered_records(tmp_path):
    source, revision = source_repo(tmp_path)
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    a = import_tactics(source, revision, serial, workers=1)
    b = import_tactics(source, revision, parallel, workers=2)
    assert a == b
    for name in ['games.jsonl', 'quarantine.jsonl', 'duplicates.jsonl']:
        assert (serial / name).read_bytes() == (parallel / name).read_bytes()


@pytest.mark.parametrize('mode', ['120000', '100755'])
def test_source_inventory_refuses_non_regular_or_executable_pgn_entries(mode):
    raw = (mode + ' blob ' + '0' * 40 + '\tDataset/中局/test.pgn\0').encode()
    with pytest.raises(ValueError, match='ordinary pinned'):
        source_inventory(raw)


def test_wrong_pinned_revision_never_creates_output(tmp_path):
    source, _ = source_repo(tmp_path)
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='pinned revision'):
        import_tactics(source, '0' * 40, output)
    assert not output.exists()
