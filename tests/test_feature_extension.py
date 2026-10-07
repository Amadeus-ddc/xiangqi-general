import json

import pytest
import torch

from xqgeneral.evidence import history_key
from xqgeneral.extend_features import extend_cache, recorded_histories
from xqgeneral.rules import START_FEN, replay
from xqgeneral.symmetry import mirror_fen, mirror_move


def cache(keys, offset):
    return {'keys': keys, 'depths': [0, 19],
            'features': [torch.full((len(keys), 90, 512), offset + i, dtype=torch.float16) for i in range(2)],
            'wdl': torch.full((len(keys), 3), float(offset), dtype=torch.float32)}


def test_extension_preserves_base_and_every_shard_value():
    base, left, right = cache(['old-a', 'old-b'], 1), cache(['new-a'], 3), cache(['new-b', 'new-c'], 5)
    actual = extend_cache(base, [left, right], [0, 19])
    assert actual['keys'] == ['old-a', 'old-b', 'new-a', 'new-b', 'new-c']
    for before, after in zip(base['features'], actual['features']):
        assert torch.equal(before, after[:2])
    assert torch.equal(actual['features'][0][2:3], left['features'][0])
    assert torch.equal(actual['features'][1][3:], right['features'][1])
    assert torch.equal(actual['wdl'][:2], base['wdl'])
    assert torch.equal(actual['wdl'][3:], right['wdl'])
    assert base['keys'] == ['old-a', 'old-b']


def test_extension_rejects_overlap_and_incompatible_feature_contracts():
    base = cache(['old-a'], 1)
    with pytest.raises(ValueError, match='overlapping'):
        extend_cache(base, [cache(['old-a'], 2)], [0, 19])
    wrong = cache(['new-a'], 2)
    wrong['depths'] = [0, 18]
    with pytest.raises(ValueError, match='depth'):
        extend_cache(base, [wrong], [0, 19])
    wrong = cache(['new-a'], 2)
    wrong['features'][0] = wrong['features'][0].float()
    with pytest.raises(ValueError, match='FP16'):
        extend_cache(base, [wrong], [0, 19])


def test_recorded_features_keep_complete_history_and_actual_color_counterpart(tmp_path):
    moves = ['h2e2', 'h7e7']
    history = replay(START_FEN, moves)
    path = tmp_path / 'roots.jsonl'
    row = {'initial_fen': START_FEN, 'moves': moves, 'history': history,
           'fen': history[-1], 'feature_key': history_key(history)}
    path.write_text(json.dumps(row) + '\n')
    actual = list(recorded_histories(path))
    native_mirror = replay(mirror_fen(START_FEN), [mirror_move(move) for move in moves])
    assert actual == [history, native_mirror]
    assert history_key(native_mirror) != history_key([mirror_fen(fen) for fen in history])
    assert [fen.split()[:5] for fen in native_mirror] == [mirror_fen(fen).split()[:5] for fen in history]
    row['feature_key'] = 'wrong-context'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='full-history'):
        list(recorded_histories(path))
