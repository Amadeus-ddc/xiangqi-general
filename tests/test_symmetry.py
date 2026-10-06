from xqgeneral.curriculum_data import make_records, verify_splits
from xqgeneral.rules import START_FEN, legal_moves, piece_map, piece_name, play
from xqgeneral.symmetry import mirror_fen, mirror_move, mirror_text, mirrored_qa


def test_color_mirror_commutes_with_legal_moves_and_preserves_task_answers():
    fen = play(START_FEN, 'b0c2')
    reflected = mirror_fen(fen)
    assert set(legal_moves(reflected)) == {mirror_move(m) for m in legal_moves(fen)}
    assert mirror_fen(mirror_fen(fen)) == fen
    assert mirror_text(mirror_text('红马 b0c2，第2行黑卒。')) == '红马 b0c2，第2行黑卒。'
    rows = make_records(games=6, seed=7, plies=16)
    assert {r['fen'].split()[1] for r in rows} == {'w', 'b'}
    for row in rows:
        mirrored = mirrored_qa(row)
        assert mirrored['split'] == row['split'] and mirrored['game_id'] == row['game_id']
        # Future questions refer to the board after the supplied continuation.
        # The root board can have an empty square at that same coordinate.
        actual = mirrored['fen']
        original = row['fen']
        for move in mirrored['future_moves']:
            actual = play(actual, move)
        for move in row['future_moves']:
            original = play(original, move)
        if row['task_type'] in {'piece', 'empty'}:
            symbol = piece_map(actual).get(mirrored['query']['square'])
            original_symbol = piece_map(original).get(row['query']['square'])
            assert mirrored['answer'] == piece_name(symbol)
            assert symbol == (original_symbol.swapcase() if original_symbol else None)
        elif row['task_type'] in {'legal', 'illegal'}:
            expected = '合法' if mirrored['query']['move'] in legal_moves(actual) else '不合法'
            assert mirrored['answer'] == expected
        elif row['task_type'] == 'locate':
            expected = sorted(s for s, p in piece_map(actual).items() if p == mirrored['query']['symbol'])
            assert mirrored['answer'] == (' '.join(expected) or '无')
    assert verify_splits(rows)['game_overlap'] == 0


def test_mirror_preserves_all_future_branch_positions():
    from xqgeneral.engine_selfplay import context_positions
    from xqgeneral.evidence import position_key
    row = make_records(games=6, seed=7, plies=16)[0]
    row = dict(row, fen=START_FEN, initial_fen=START_FEN, moves=[], history=[START_FEN],
               future_moves=['b0c2'], future_branches=[['b0c2', 'b9c7'], ['h0g2', 'h9g7']])
    mirror = mirrored_qa(row)
    assert mirror['future_branches'] == [[mirror_move(m) for m in line] for line in row['future_branches']]
    original_fens = [START_FEN]
    for line in [row['future_moves'], *row['future_branches']]:
        fen = START_FEN
        for move in line:
            fen = play(fen, move); original_fens.append(fen)
    assert context_positions(mirror) == {position_key(mirror_fen(f)) for f in original_fens}
    assert mirrored_qa(mirror)['future_branches'] == row['future_branches']


def test_mirror_transforms_explicit_rank_names_in_teacher_prose():
    original = '红兵由e6横到d6，始终在rank6横线上；黑卒在rank3的f3、g3间活动。'
    mirrored = mirror_text(original)
    assert mirrored == '黑卒由e3横到d3，始终在rank3横线上；红兵在rank6的f6、g6间活动。'
    assert mirror_text(mirrored) == original
    assert mirror_text('rank10不是单个棋盘横线。') == 'rank10不是单个棋盘横线。'
