import random

import pytest

from xqgeneral.evidence import history_key, position_key
from xqgeneral.rules import START_FEN, adjudicate, gives_check, legal_moves, piece_map, play, replay
from xqgeneral.selfplay_grounding import grounding_candidates, grounding_pair, question_variant
from xqgeneral.symmetry import mirror_fen


def query(moves=(), pv=('b0c2', 'b9c7', 'h0g2', 'h9g7', 'a0a1', 'a9a8')):
    history = replay(START_FEN, moves)
    row = {'id': 'root', 'feature_key': history_key(history), 'initial_fen': START_FEN,
           'moves': list(moves), 'history': history, 'fen': history[-1], 'split': 'train',
           'game_id': 'engine-selfplay-training', 'provenance': 'engine_generated_game'}
    return {'id': 'root-query', 'feature_key': row['feature_key'], 'record': row,
            'oracle': {'best_move': pv[0], 'candidates': [{'move': pv[0], 'pv': list(pv)}]}}


def test_grounding_keeps_histories_and_answers_rules_on_the_requested_future_board():
    q = query()
    rows, reason = grounding_pair(q, random.Random(12))
    assert reason is None and len(rows) == 44
    assert {r['stage'] for r in rows} == {'static_current', 'dynamic_current', 'static_future', 'dynamic_future'}
    original = [r for r in rows if 'augmentation_parent' not in r]
    assert all(r['history'] == q['record']['history'] and r['fen'] == START_FEN for r in original)
    for row in original:
        target = row['fen']
        for move in row['future_moves']:
            target = play(target, move)
        assert not adjudicate(row['initial_fen'], row['moves'] + row['future_moves'])['ended']
        if row['task_type'] in {'legal', 'illegal'}:
            valid = row['query']['move'] in legal_moves(target)
            assert row['answer'] == ('合法' if valid else '不合法')
        if row['task_type'] == 'moves':
            assert row['answer'].split() == sorted(m for m in legal_moves(target)
                                                  if m[:2] == row['query']['source'])
    assert len({r['feature_key'] for r in rows}) == 2
    assert all(not r['teacher']['neural_prose_generated'] for r in rows)


def test_grounding_refuses_historical_terminal_futures_and_inconsistent_inputs():
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    assert grounding_pair(query(cycle * 2), random.Random(1)) == ([], 'terminal_root')
    before_end = query((cycle * 2)[:-1], ('c7b9', 'a0a0'))
    assert grounding_pair(before_end, random.Random(1)) == ([], 'no_nonterminal_future')
    bad = query()
    bad['record']['history'] = [play(START_FEN, 'h0g2')]
    with pytest.raises(ValueError, match='full history'):
        grounding_pair(bad, random.Random(1))
    heldout = query()
    heldout['record']['split'] = 'validation'
    with pytest.raises(ValueError, match='training games'):
        grounding_pair(heldout, random.Random(1))


def test_grounding_excludes_deep_future_and_mirrored_reserved_positions():
    q = query()
    seed = 12
    rows, _ = grounding_pair(q, random.Random(seed))
    future_row = next(r for r in rows if r['stage'] == 'dynamic_future' and 'augmentation_parent' not in r)
    target = q['record']['fen']
    for move in future_row['future_moves']:
        target = play(target, move)
    assert position_key(target) != position_key(START_FEN)
    for reserved in [{position_key(target)}, {position_key(mirror_fen(target))}]:
        assert grounding_pair(q, random.Random(seed), reserved) == ([], 'heldout_root_or_future')
    assert grounding_pair(q, random.Random(seed), available_keys={q['feature_key']}) == (
        [], 'root_missing_from_verified_policy_data')


@pytest.mark.parametrize('max_future_plies,question_formats', [(6, 1), (8, 3)])
def test_parallel_grounding_preserves_order_labels_and_terminal_rejections(max_future_plies, question_formats):
    import json
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    queries = [query(), query(['b0c2'], ('b9c7', 'h0g2', 'h9g7', 'a0a1')),
               query(cycle * 2), query((cycle * 2)[:-1], ('c7b9', 'a0a0'))]
    for index, q in enumerate(queries):
        q['id'] = f'query-{index}'
    options = dict(max_future_plies=max_future_plies, question_formats=question_formats)
    serial = list(grounding_candidates(queries, frozenset(), None, 42, workers=1, **options))
    parallel = list(grounding_candidates(queries, frozenset(), None, 42, workers=2, chunk_size=1, **options))
    assert json.dumps(serial, ensure_ascii=False) == json.dumps(parallel, ensure_ascii=False)
    assert [reason for _, _, reason in serial] == [None, None, 'terminal_root', 'no_nonterminal_future']


def test_eight_ply_future_checks_every_prefix_including_mirror():
    q = query(pv=('b0c2', 'b9c7', 'h0g2', 'h9g7', 'a0a1', 'a9a8', 'a1a2', 'a8a7'))

    class LongHorizon(random.Random):
        def randrange(self, start, stop=None, step=1):
            return 8 if (start, stop, step) == (1, 9, 1) else super().randrange(start, stop, step)

    rows, reason = grounding_pair(q, LongHorizon(12), max_future_plies=8, question_formats=3)
    assert reason is None and len(rows) == 44
    future = next(r for r in rows if r['stage'] == 'dynamic_future' and 'augmentation_parent' not in r)
    assert future['future_moves'] == q['oracle']['candidates'][0]['pv']
    seventh = replay(START_FEN, future['future_moves'][:7])[-1]
    for reserved in ({position_key(seventh)}, {position_key(mirror_fen(seventh))}):
        assert grounding_pair(q, LongHorizon(12), reserved, max_future_plies=8, question_formats=3) == (
            [], 'heldout_root_or_future')
    for row in rows:
        target = replay(row['initial_fen'], row['moves'] + row['future_moves'])[-1]
        assert not adjudicate(row['initial_fen'], row['moves'] + row['future_moves'])['ended']
        legal, board = legal_moves(target), piece_map(target)
        if row['task_type'] == 'captures':
            assert row['answer'] == (' '.join(sorted(m for m in legal if m[2:] in board)) or '无')
        if row['task_type'] == 'checks':
            assert row['answer'] == (' '.join(sorted(m for m in legal if gives_check(target, m))) or '无')


def test_question_variants_keep_native_answers_and_valid_mirror_lineage():
    q, seed = query(), 12
    plain, _ = grounding_pair(q, random.Random(seed))
    varied, reason = grounding_pair(q, random.Random(seed), question_formats=3)
    assert reason is None and len(varied) == len(plain)
    assert {r['question_variant'] for r in varied} == {0, 1, 2}
    assert len({r['id'] for r in varied}) == len(varied)
    originals = {r['id']: r for r in varied if 'augmentation_parent' not in r}
    for before, after in zip(plain, varied, strict=True):
        for field in ('answer', 'query', 'future_moves', 'history', 'feature_key', 'split'):
            assert before[field] == after[field]
        assert after['question'] == question_variant(before, after['question_variant'])
        if 'augmentation_parent' in after:
            parent = originals[after['augmentation_parent']]
            assert after['question_variant'] == parent['question_variant']
        if after['future_moves']:
            assert after['question'].startswith("依次走 " + ' '.join(after['future_moves']) + ' 后，')
        for variant in (1, 2):
            assert question_variant(before, variant) != before['question']


@pytest.mark.parametrize('options', [{'max_future_plies': 9}, {'max_future_plies': 0},
                                   {'max_future_plies': True}, {'question_formats': 2}])
def test_grounding_refuses_unsupported_horizon_or_format(options):
    with pytest.raises(ValueError):
        grounding_pair(query(), random.Random(12), **options)
