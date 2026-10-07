import pytest

from xqgeneral.curriculum_data import make_records, verify_splits
from xqgeneral.rules import START_FEN, play


def record(split, game, fen=START_FEN, future=(), branches=()):
    return {'split': split, 'game_id': game, 'fen': fen,
            'future_moves': list(future), 'future_branches': [list(x) for x in branches]}


def test_parallel_split_check_matches_real_seeded_curriculum():
    rows = make_records(games=5, plies=20, seed=20261041)
    assert rows
    assert verify_splits(iter(rows), workers=2) == verify_splits(rows)


def test_duplicate_future_context_keeps_every_game_owner():
    other = play(START_FEN, 'h0g2')
    rows = [record('train', 'first'), record('train', 'shared'),
            record('validation', 'shared', other)]
    with pytest.raises(ValueError, match='Data leakage'):
        verify_splits(rows, workers=2)


def test_parallel_check_rejects_late_alternative_branch_overlap():
    late = play(play(START_FEN, 'b0c2'), 'b9c7')
    rows = [record('train', 'train-game', future=['h0g2'],
                   branches=[['b0c2', 'b9c7']]),
            record('validation', 'heldout-game', late)]
    with pytest.raises(ValueError, match='Data leakage'):
        verify_splits(rows, workers=2)


def test_parallel_check_propagates_illegal_unused_future():
    with pytest.raises(ValueError, match='Illegal move'):
        verify_splits([record('train', 'game', branches=[['a0a0']])], workers=2)


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
def test_split_worker_count_requires_positive_integer(workers):
    with pytest.raises(ValueError, match='positive integer'):
        verify_splits([], workers=workers)
