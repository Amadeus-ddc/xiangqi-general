"""Actual supplied-line endings are distinct from archived mate estimates."""
import copy
import hashlib
import json

import pytest

from xqgeneral import prose_moves
from xqgeneral.rules import START_FEN, in_check, legal_moves, play
from test_grounded_consolidation_evidence import packet
from test_search_child_prompts import CapturingPredictor, make_search
from test_search_distillation import root_record


PROFILE = 'grounded_terminal_child_prose_v3'
RED_FEN = '4k4/1R3R3/9/9/4P4/9/9/9/9/4K4 w - - 0 1'
BLACK_FEN = '4k4/9/9/9/9/4p4/9/9/3r3r1/4K4 b - - 0 1'


def render(fen, moves):
    content, target = packet(fen, moves)
    target['evaluation'] = {'type': 'cp', 'value': 17, 'perspective': 'side_to_move'}
    target['branches'][0]['evaluation']['value'] = 13
    original = copy.deepcopy(content)
    messages = prose_moves.consolidation_messages(
        content, fen, target, consolidation_profile=PROFILE)
    assert content == original
    return messages, json.loads(messages[-1]['content'])


@pytest.mark.parametrize('fen,move,result,winner,checked', [
    (RED_FEN, 'b8b9', 'checkmate', 'red', True),
    (BLACK_FEN, 'h1h0', 'checkmate', 'black', True),
    (RED_FEN, 'b8d8', 'stalemate_loss', 'red', False),
])
def test_actual_terminal_endpoint_even_when_reference_score_is_cp(
        fen, move, result, winner, checked):
    after = play(fen, move)
    assert not legal_moves(after) and in_check(after) is checked
    messages, payload = render(fen, (move,))
    facts = payload['native_coaching_evidence']
    assert len(facts['lines']) == 1
    ending = facts['lines'][0]['line_end']
    assert ending == {
        'fen': after, 'side_to_move': '黑方' if winner == 'red' else '红方',
        'in_check': checked, 'legal_move_count': 0,
        'position_result': result, 'winner': winner,
    }
    # Actual line endings must neither replace nor reinterpret the search scores.
    assert facts['root_oracle_evaluation']['value'] == 17
    assert facts['branch_estimates_root_perspective'][0]['evaluation']['value'] == 13
    assert facts['line_ending_scope'] == 'supplied_line_endpoint_without_pre_root_history'
    assert '困毙' in messages[0]['content'] and '不是和棋' in messages[0]['content']
    assert '只写将军' in messages[0]['content']
    assert prose_moves.checked_consolidation_payload(payload, PROFILE)['children_with_prose'] == 1


def test_nonterminal_endpoint_does_not_invent_a_winner_or_mate():
    _, payload = render(START_FEN, ('b0c2', 'b9c7'))
    ending = payload['native_coaching_evidence']['lines'][0]['line_end']
    assert ending['position_result'] == 'ongoing' and ending['winner'] is None
    assert ending['legal_move_count'] == len(legal_moves(ending['fen'])) > 0


@pytest.mark.parametrize('option', [None, 0, 1])
def test_terminal_evidence_option_is_not_an_untyped_profile_alias(option):
    _, target = packet(RED_FEN, ('b8b9',))
    with pytest.raises(ValueError, match='boolean option'):
        prose_moves.native_coaching_evidence(RED_FEN, target, include_line_endings=option)


@pytest.mark.parametrize('field,value', [
    ('legal_move_count', 1), ('in_check', False),
    ('legal_move_count', False), ('in_check', 1),
    ('legal_move_count', 0.0), ('in_check', 1.0),
    ('position_result', 'ongoing'), ('winner', 'black'), ('fen', START_FEN),
])
def test_terminal_fact_tampering_fails_before_teacher_generation(field, value):
    _, payload = render(RED_FEN, ('b8b9',))
    payload['native_coaching_evidence']['lines'][0]['line_end'][field] = value
    with pytest.raises(ValueError, match='differs from native facts'):
        prose_moves.checked_consolidation_payload(payload, PROFILE)


def test_missing_terminal_fact_or_relabelled_legacy_evidence_is_rejected():
    _, payload = render(RED_FEN, ('b8b9',))
    del payload['native_coaching_evidence']['lines'][0]['line_end']
    with pytest.raises(ValueError, match='differs from native facts'):
        prose_moves.checked_consolidation_payload(payload, PROFILE)
    content, target = packet(RED_FEN, ('b8b9',))
    old = json.loads(prose_moves.consolidation_messages(
        content, RED_FEN, target, consolidation_profile='grounded_child_prose_v2')[-1]['content'])
    old['consolidation_profile'] = PROFILE
    with pytest.raises(ValueError, match='differs from native facts'):
        prose_moves.checked_consolidation_payload(old, PROFILE)


def test_grounded_v2_prompt_remains_byte_identical_and_has_no_new_fields():
    content, target = packet()
    messages = prose_moves.consolidation_messages(
        content, content['root_fen'], target, consolidation_profile='grounded_child_prose_v2')
    assert hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest() == (
        '45b6d588100f8314672e3e72f0f3832b1364037170b7999c4e4289d3bee91388')
    facts = json.loads(messages[-1]['content'])['native_coaching_evidence']
    assert 'line_ending_scope' not in facts and 'line_end' not in facts['lines'][0]


def test_terminal_profile_retains_actual_child_prose_and_miner_trace():
    original = make_search(CapturingPredictor(), 'paper_recursive',
                           consolidation_profile='grounded_child_prose_v2').mine(root_record())
    query, trace, reason = make_search(CapturingPredictor(), 'paper_recursive',
                                      consolidation_profile=PROFILE).mine(root_record())
    assert (trace, reason) == (original[1], original[2])
    assert query['target_fields'] == original[0]['target_fields']
    old, new = (json.loads(q['messages'][-1]['content']) for q in (original[0], query))
    assert new['child_analyses'] == old['child_analyses']
    assert new['root_line_facts'] == old['root_line_facts']
    assert all('line_end' in line for line in new['native_coaching_evidence']['lines'])


def test_terminal_evidence_does_not_substitute_for_missing_child_prose():
    content, target = packet()
    content['child_analyses'][0]['unverified_explanation'] = ''
    with pytest.raises(prose_moves.MissingChildProseError):
        prose_moves.consolidation_messages(
            content, content['root_fen'], target, consolidation_profile=PROFILE)


def test_numeric_child_profile_is_rejected_before_model_work():
    predictor = CapturingPredictor()
    with pytest.raises(ValueError, match='full_explanation'):
        make_search(predictor, 'paper_recursive', consolidation_profile=PROFILE,
                    child_prompt_profile='numeric_six_ply_v1')
    assert not predictor.calls
