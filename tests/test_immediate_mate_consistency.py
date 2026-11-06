"""Native terminal witnesses and collection policy boundaries; no model inference."""
from copy import deepcopy
import json

import pytest

from test_search_collection import TEACHER, query, response
from test_search_pilot import ControlledChain, setup_pilot
from xqgeneral import collect_search, search_pilot
from xqgeneral.evidence import atomic_json, history_key
from xqgeneral.explanations import (IMMEDIATE_MATE_CONSISTENCY_PROFILE,
                                  immediate_mate_score_consistency, move_facts)
from xqgeneral.rules import START_FEN, adjudicate, in_check, play


NEGATIVE_ROOT = '9/9/1N2k4/7R1/4R1b2/9/9/9/5K3/9 b - - 0 1'
POSITIVE_ROOT = '4k4/9/5r3/9/9/9/3n5/5A3/5K3/9 b - - 0 1'
STALEMATE_ROOT = '4k4/3R5/9/5R3/9/9/9/9/9/5K3 w - - 0 1'


def mate_query(fen, pv, score):
    evaluation = {'type': 'mate', 'value': score, 'perspective': 'side_to_move'}
    return {'id': 'q-search', 'source_root_id': 'q', 'depth': 0,
            'record': {'id': 'q', 'fen': fen, 'initial_fen': fen, 'moves': [],
                       'history': [fen], 'feature_key': history_key([fen]),
                       'game_id': 'controlled-native-mate', 'split': 'train',
                       'provenance': 'handcrafted CPU terminal witness'},
            'target_fields': {'move': pv[0], 'pv': list(pv), 'candidates': [pv[0]],
                'evaluation': evaluation, 'facts': move_facts(fen, pv[0]),
                'branches': [{'move': pv[0], 'pv': list(pv), 'evaluation': evaluation}]}}


def neutral_response():
    raw = response()
    raw['text'] = json.dumps({'explanation': '只依据给定棋盘、原始历史与主变化解释建议，'
        '逐步核对棋子、吃子和将军，分清行棋方与对手的评分视角。'
        '没有提供的后续不当作已经发生，不能只凭措辞流畅就认定强制结果；'
        '备选的收益与风险也须与实际给出的变化一致。'}, ensure_ascii=False)
    return raw


def test_legal_check_evasion_does_not_realize_the_declared_mate_minus_one():
    q = mate_query(NEGATIVE_ROOT, ['e7d7', 'e5d5'], -1)
    before = deepcopy(q)
    assert not adjudicate(NEGATIVE_ROOT, q['target_fields']['pv'])['ended']
    assert adjudicate(NEGATIVE_ROOT, ['e7d7', 'h6d6'])['winner'] == 'red'
    proof = immediate_mate_score_consistency(q['record'], q['target_fields'])
    assert not proof['valid'] and proof['checked_scored_lines'] == 2
    assert proof['profile'] == IMMEDIATE_MATE_CONSISTENCY_PROFILE
    assert not proof['strategic_prose_verified'] and q == before


@pytest.mark.parametrize('fen,pv,score', [
    (NEGATIVE_ROOT, ['e7d7', 'h6d6'], -1),
    (POSITIVE_ROOT, ['f7f2'], 1),
    (STALEMATE_ROOT, ['f6f7'], 1),
])
def test_actual_terminal_wins_accept_both_colors_and_xiangqi_stalemate(fen, pv, score):
    q = mate_query(fen, pv, score)
    assert immediate_mate_score_consistency(q['record'], q['target_fields'])['valid']
    if fen == STALEMATE_ROOT:
        assert not in_check(play(fen, pv[0]))
    assert consolidated(q, True) == consolidated(q, False)


def consolidated(q, required):
    return collect_search.consolidated_label(q, neutral_response(), TEACHER,
        require_immediate_mate_consistency=required)


def test_collection_is_compatible_by_default_and_never_repairs_bad_pv():
    q = mate_query(NEGATIVE_ROOT, ['e7d7', 'e5d5'], -1)
    before = deepcopy(q)
    legacy = collect_search.consolidated_label(q, neutral_response(), TEACHER)
    assert legacy == consolidated(q, False)
    with pytest.raises(ValueError, match='immediate-mate consistency'):
        consolidated(q, True)
    accepted, rejected = collect_search.rebuild_labels([q], [neutral_response()], TEACHER,
        set(), require_immediate_mate_consistency=True)
    assert not accepted and [r['id'] for r in rejected] == [q['id']]
    assert q == before


def test_nonselected_branch_mate_is_checked_as_well_as_the_root_score():
    q = deepcopy(query())
    q['target_fields']['branches'][1]['evaluation'] = {
        'type': 'mate', 'value': 1, 'perspective': 'side_to_move'}
    assert consolidated(q, False)
    with pytest.raises(ValueError, match='branch_1'):
        consolidated(q, True)


def test_missing_original_history_and_incomplete_negative_mate_line_fail_clearly():
    q = mate_query(NEGATIVE_ROOT, ['e7d7'], -1)
    assert not immediate_mate_score_consistency(q['record'], q['target_fields'])['valid']
    q = mate_query(POSITIVE_ROOT, ['f7f2'], 1)
    del q['record']['initial_fen']
    assert not immediate_mate_score_consistency(q['record'], q['target_fields'])['valid']


def test_mismatched_root_and_already_terminal_original_history_are_not_accepted():
    q = mate_query(POSITIVE_ROOT, ['f7f2'], 1)
    q['record']['fen'] = START_FEN
    assert not immediate_mate_score_consistency(q['record'], q['target_fields'])['valid']
    q['record'].update(fen=play(POSITIVE_ROOT, 'f7f2'), moves=['f7f2'])
    assert not immediate_mate_score_consistency(q['record'], q['target_fields'])['valid']


def test_centipawn_or_longer_mate_estimates_are_outside_this_narrow_check():
    q = mate_query(NEGATIVE_ROOT, ['e7d7', 'e5d5'], -1)
    for value in [0, -2, 3]:
        q['target_fields']['evaluation'] = {'type': 'cp' if value == 0 else 'mate',
                                           'value': value, 'perspective': 'side_to_move'}
        q['target_fields']['branches'] = []
        result = immediate_mate_score_consistency(q['record'], q['target_fields'])
        assert result['valid'] and result['checked_scored_lines'] == 0
        assert result['strategic_prose_verified'] is False


@pytest.mark.parametrize('required', [None, 0, 1, 'true', []])
def test_requirement_is_boolean_and_invalid_modes_do_not_become_label_rejections(required):
    with pytest.raises(ValueError, match='must be boolean'):
        consolidated(query(), required)
    with pytest.raises(ValueError, match='must be boolean'):
        collect_search.rebuild_labels([], [], TEACHER, set(),
            require_immediate_mate_consistency=required)


@pytest.mark.parametrize('required', ['absent', False, True])
def test_pilot_passes_optional_policy_to_the_real_collection_cli_and_resume_contract(
        tmp_path, monkeypatch, required):
    pipeline, validation, config, q, _ = setup_pilot(tmp_path, monkeypatch)
    if required != 'absent':
        settings = json.loads(config.read_text())
        settings['require_immediate_mate_consistency'] = required
        atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, q)
    output = tmp_path / 'pilot'
    result = search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    command = next(c for c in chain.commands if 'xqgeneral.collect_search' in c)
    proof = json.loads((output / 'labels/manifest.json').read_text())
    enabled = required is True
    assert ('--require-immediate-mate-consistency' in command) is enabled
    assert ('require_immediate_mate_consistency' in proof['config']) is enabled
    assert proof['verification'].get('immediate_mate_score_consistency_required', False) is enabled
    if enabled:
        assert proof['verification']['immediate_mate_consistency_profile'] == IMMEDIATE_MATE_CONSISTENCY_PROFILE
    else:
        assert 'immediate_mate_consistency_profile' not in proof['verification']
    assert result['accepted_structured_training_labels'] == 1
    assert not result['strategic_prose_fully_verified']
    assert result == search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output, resume=True)
    assert len(chain.commands) == 3


@pytest.mark.parametrize('required', [None, 0, 1, 'true'])
def test_bad_pilot_policy_fails_before_model_loading(tmp_path, monkeypatch, required):
    pipeline, validation, config, q, _ = setup_pilot(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    settings['require_immediate_mate_consistency'] = required
    atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, q)
    with pytest.raises(ValueError, match='must be boolean'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert not chain.commands and chain.student_loads == chain.teacher_loads == 0
