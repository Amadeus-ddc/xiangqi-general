import json
from pathlib import Path

import pytest

from xqgeneral.evidence import history_key
from xqgeneral.foundation_preflight import clean_recipe, native_answer, native_context, validate_task
from xqgeneral.rules import START_FEN, replay


def context():
    moves = ['h2e2', 'h7e7']
    history = replay(START_FEN, moves)
    return {'initial_fen': START_FEN, 'moves': moves, 'history': history,
            'fen': history[-1], 'feature_key': history_key(history)}


def test_preflight_requires_fresh_latent_bridge_recipe():
    recipe = json.loads((Path(__file__).parents[1] / 'configs/foundation-human-engine-clean-v2.json').read_text())
    clean_recipe(recipe)
    with pytest.raises(ValueError, match='literal board'):
        clean_recipe(dict(recipe, board_text='dictionary'))
    with pytest.raises(ValueError, match='inherit'):
        clean_recipe(dict(recipe, init_from='old-learned-course.pt'))


def test_preflight_native_context_rejects_wrong_history_and_terminal_recorded_roots():
    row = context()
    assert native_context(row) == row['feature_key']
    with pytest.raises(ValueError, match='history'):
        native_context(dict(row, feature_key='incorrect-fullmove-history'))
    moves = ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2
    history = replay(START_FEN, moves)
    with pytest.raises(ValueError, match='terminal'):
        native_context({'initial_fen': START_FEN, 'moves': moves, 'history': history,
                        'fen': history[-1], 'feature_key': history_key(history),
                        'recorded_source_kind': 'recorded_human_match'})


def test_native_answer_replays_actual_future_without_using_stored_supervision():
    row = dict(context(), future_moves=['b0c2'], task_type='piece', query={'square': 'c2'}, answer='wrong')
    assert native_answer(row) == '红马'
    row.update(task_type='empty', query={'square': 'b0'})
    assert native_answer(row) == '空'
    row.update(task_type='legal', query={'move': 'b9c7'})
    assert native_answer(row) == '合法'


def test_independent_terminal_probes_never_enter_training_or_validation_contract():
    row = dict(context(), split='test', stage='dynamic_current', task_type='terminal',
               future_moves=[], query={}, answer='有')
    validate_task(row)
    for split in ('train', 'validation'):
        with pytest.raises(ValueError, match='QA contract'):
            validate_task(dict(row, split=split))
    with pytest.raises(ValueError, match='native rules'):
        validate_task(dict(row, answer='无'))
