from copy import deepcopy

import pytest
import torch

from xqgeneral.evidence import history_key, position_key
from xqgeneral.foundation_readback import checked_preserved_cache, checked_question_contexts, checked_recorded_roots
from xqgeneral.human_games import assigned_split
from xqgeneral.rules import START_FEN, replay
from xqgeneral.symmetry import mirror_fen, mirror_move


SEED = 91


def recorded_fixture(split='train', ply=1):
    game_id = next(f'recorded-fixture-{i}' for i in range(1000)
                   if assigned_split(f'recorded-fixture-{i}', SEED) == split)
    moves = ['h2e2', 'h7e7', 'b0c2', 'b9c7']
    history = replay(START_FEN, moves)
    source = {'game_id': game_id, 'split': split, 'initial_fen': START_FEN,
              'moves': moves, 'history': history, 'source_kind': 'recorded_human_match',
              'source': {'category': 'recorded_human_match', 'blob_sha1': 'source-fixture'}}
    prefix = history[:ply + 1]
    root = {'game_id': game_id, 'split': split, 'ply': ply, 'initial_fen': START_FEN,
            'moves': moves[:ply], 'history': prefix, 'fen': prefix[-1],
            'feature_key': history_key(prefix), 'future_moves': moves[ply:ply + 2],
            'recorded_source': source['source'], 'recorded_source_kind': source['source_kind']}
    mirror_moves = [mirror_move(m) for m in root['moves']]
    mirrored = replay(mirror_fen(START_FEN), mirror_moves)
    mirror = {'initial_fen': mirror_fen(START_FEN), 'moves': mirror_moves,
              'history': mirrored, 'fen': mirrored[-1], 'feature_key': history_key(mirrored),
              'recorded_source_kind': source['source_kind']}
    contexts = {(START_FEN, tuple(root['moves'])): root,
                (mirror['initial_fen'], tuple(mirror_moves)): mirror}
    keys = {root['feature_key'], mirror['feature_key']}
    return root, source, contexts, keys


def verify(root, source, contexts, keys, prior=None):
    return checked_recorded_roots([root], {source['game_id']: source}, contexts, keys,
                                  prior or {'train': set(), 'heldout': set()}, SEED)


def test_readback_requires_exact_original_game_prefix_future_and_source():
    root, source, contexts, keys = recorded_fixture()
    result = verify(root, source, contexts, keys)
    assert result['roots_by_split'] == {'train': 1}
    assert result['all_original_source_prefixes_and_futures_match']
    for field, value in [('future_moves', ['b9c7']), ('recorded_source', {'category': 'another-source'}),
                         ('split', 'validation')]:
        with pytest.raises(ValueError, match='original source'):
            verify(dict(root, **{field: value}), source, contexts, keys)


def test_readback_rejects_second_future_and_color_counterpart_holdout_leaks():
    root, source, contexts, keys = recorded_fixture()
    second_future = source['history'][root['ply'] + 2]
    for heldout in (second_future, mirror_fen(second_future)):
        with pytest.raises(ValueError, match='training overlaps'):
            verify(root, source, contexts, keys,
                   {'train': set(), 'heldout': {position_key(heldout)}})
    root, source, contexts, keys = recorded_fixture('validation')
    with pytest.raises(ValueError, match='prior training'):
        verify(root, source, contexts, keys,
               {'train': {position_key(mirror_fen(source['history'][3]))}, 'heldout': set()})


def test_readback_rejects_new_validation_test_future_overlap():
    validation, val_source, val_contexts, val_keys = recorded_fixture('validation', 1)
    test, test_source, test_contexts, test_keys = recorded_fixture('test', 2)
    with pytest.raises(ValueError, match='validation and test'):
        checked_recorded_roots([validation, test],
                              {val_source['game_id']: val_source, test_source['game_id']: test_source},
                              val_contexts | test_contexts, val_keys | test_keys,
                              {'train': set(), 'heldout': set()}, SEED)


def test_readback_rejects_duplicate_or_inconsistent_question_history_keys():
    _, _, contexts, keys = recorded_fixture()
    values = list(contexts.values())
    observed, recorded = checked_question_contexts(values, keys, 2)
    assert observed == keys and len(recorded) == 2
    with pytest.raises(ValueError, match='identity'):
        checked_question_contexts(values + values[:1], keys, 2)
    with pytest.raises(ValueError, match='identity'):
        checked_question_contexts([dict(values[0], feature_key=values[1]['feature_key'])], keys, 1)
    with pytest.raises(ValueError, match='coverage'):
        checked_question_contexts(values, keys, 3)


@pytest.mark.parametrize('changed', ['order', 'feature', 'wdl', 'dtype'])
def test_readback_preserves_all_old_cache_values_and_dtype(changed):
    old = {'keys': ['old-a', 'old-b'], 'depths': [0, 1],
           'features': [torch.arange(8, dtype=torch.float16).reshape(2, 2, 2)] * 2,
           'wdl': torch.tensor([[.1, .2, .7], [.5, .3, .2]])}
    cache = {'keys': ['old-a', 'old-b', 'new'], 'depths': [0, 1],
             'features': [torch.cat([f, torch.zeros_like(f[:1])]) for f in old['features']],
             'wdl': torch.cat([old['wdl'], torch.zeros_like(old['wdl'][:1])])}
    assert checked_preserved_cache(cache, old) == 2
    changed_cache = deepcopy(cache)
    if changed == 'order':
        changed_cache['keys'][:2] = ['old-b', 'old-a']
    elif changed == 'feature':
        changed_cache['features'][1][1, 1, 1] += 1
    elif changed == 'wdl':
        changed_cache['wdl'][1, 2] += .1
    else:
        changed_cache['features'][0] = changed_cache['features'][0].float()
    with pytest.raises(ValueError, match='changed'):
        checked_preserved_cache(changed_cache, old)
