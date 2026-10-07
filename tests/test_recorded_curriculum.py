from collections import Counter
import copy
import json

import pytest
from xqgeneral.human_games import assigned_split, parse_game
from xqgeneral.recorded_curriculum import question_groups, recorded_pair, root_candidates
from xqgeneral.curriculum_data import verify_splits
from xqgeneral.engine_selfplay import context_positions
from xqgeneral.evidence import position_key
from xqgeneral.rules import play


def recorded_game():
    game = parse_game(('[Game "Chinese Chess"]\n[Date "2021-01-01"]\n[Red "棋手甲"]\n'
                       '[Black "棋手乙"]\n[Result "1/2-1/2"]\n\n'
                       '1. H2-E2 H9-G7 2. H0-G2 I9-H9 3. B0-C2 B7-E7 1/2-1/2').encode())
    game.update(split=assigned_split(game['game_id'], 20261051), players=['棋手甲', '棋手乙'],
                source_kind='recorded_human_match', provenance='recorded_source',
                source={'content_sha256': 'fixture-only', 'declared_license': 'test_fixture'})
    return game


def test_recorded_future_answers_use_the_actual_game_and_mirrors_are_derived():
    game = recorded_game()
    root, footprint = next(root_candidates(game, seed=7, min_ply=1))
    rows = recorded_pair(root, seed=7)
    assert len(rows) == 44
    assert Counter(r['stage'] for r in rows) == {
        'static_current': 12, 'dynamic_current': 10, 'static_future': 12, 'dynamic_future': 10}
    for row in rows:
        assert row['recorded_source_game_id'] == game['game_id']
        assert row['recorded_continuation_is_best_move_label'] is False
        assert context_positions(row) <= footprint
        if not row.get('augmentation_parent'):
            assert row['future_moves'] == root['future_moves'][:len(row['future_moves'])]
        if row['stage'] == 'static_future' and row['task_type'] == 'piece':
            fen = row['fen']
            for move in row['future_moves']:
                fen = play(fen, move)
            from xqgeneral.rules import piece_map, piece_name
            assert row['answer'] == piece_name(piece_map(fen)[row['query']['square']])
    verify_splits(rows)


def test_recorded_generation_rejects_changed_history_and_illegal_future():
    root, _ = next(root_candidates(recorded_game(), seed=7, min_ply=1))
    changed = copy.deepcopy(root)
    changed['history'][-1] = changed['history'][0]
    with pytest.raises(ValueError, match='native full history'):
        recorded_pair(changed, 7)
    changed = copy.deepcopy(root)
    changed['future_moves'] = ['a0a0']
    with pytest.raises(ValueError, match='Illegal'):
        recorded_pair(changed, 7)


def test_future_split_footprint_is_computed_from_native_recorded_moves():
    game = recorded_game()
    game['history'][2] = game['history'][0]
    with pytest.raises(ValueError, match='future history'):
        list(root_candidates(game, seed=7, min_ply=1))


def test_serial_and_spawned_recorded_question_groups_are_identical():
    roots = [root for root, _ in root_candidates(recorded_game(), seed=7, min_ply=1)]
    single = list(question_groups(roots, 7, workers=1))
    multiple = list(question_groups(roots, 7, workers=2))
    assert json.dumps(single, ensure_ascii=False) == json.dumps(multiple, ensure_ascii=False)
