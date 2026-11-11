"""Native board controls for the optional teacher input; no neural quality claim."""
import copy
import hashlib
import json

import pytest

from xqgeneral import prose_moves
from xqgeneral.rules import START_FEN, piece_map, piece_name, side
from test_consolidation_profiles import packet_and_target
from test_search_child_prompts import CapturingPredictor, make_search
from test_search_distillation import root_record


PROFILE = 'grounded_child_prose_v2'
CHECK_FEN = '2bakabr1/9/6n2/p1p1n1p1p/4C4/1NP3P2/P3P2cP/6N2/9/1RBAKABR1 b - - 0 11'
MATERIAL_FEN = '2b1ka3/4a2r1/4b4/p3R3p/4p1p2/2P6/P3P3P/4B4/9/3AKAB2 w - - 3 18'


def packet(fen=CHECK_FEN, moves=('g9e7', 'b4c6')):
    target = {'move': moves[0], 'pv': list(moves), 'candidates': [moves[0]],
              'evaluation': {'type': 'cp', 'value': -694, 'perspective': 'side_to_move'},
              'branches': [{'move': moves[0], 'pv': list(moves),
                            'evaluation': {'type': 'cp', 'value': -712,
                                           'perspective': 'side_to_move'}}]}
    content = {'root_fen': fen, 'root_side': side(fen), 'target_fields': target,
               'root_board': {s: piece_name(p) for s, p in piece_map(fen).items()},
               'child_analyses': [{'move': moves[0], 'terminal_outcome': None,
                   'evaluation': {'type': 'cp', 'value': 712, 'perspective': 'side_to_move'},
                   'unverified_explanation': '原始子讲解；未知语义，必须核对。'}]}
    return content, target


def rendered(content, target):
    return prose_moves.consolidation_messages(content, content['root_fen'], target,
                                              consolidation_profile=PROFILE)


def test_old_explicit_raw_profile_is_byte_identical():
    content, target = packet_and_target()
    messages = prose_moves.consolidation_messages(content, START_FEN, target,
                                                  consolidation_profile='raw_child_prose_v1')
    assert hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest() == (
        '0f51d55bc0e24843d76234fe2c642533f67b8f824db1703b76329516f396e120')
    assert 'native_coaching_evidence' not in json.loads(messages[-1]['content'])


def test_black_branch_already_has_root_sign_child_original_stays_opposite():
    content, target = packet()
    before = copy.deepcopy(content)
    messages = rendered(content, target)
    actual = json.loads(messages[-1]['content'])
    assert content == before
    assert actual['child_analyses'] == content['child_analyses']
    facts = actual['native_coaching_evidence']
    assert facts['root_side'] == '黑方'
    assert facts['root_oracle_evaluation']['value'] == -694
    assert facts['branch_estimates_root_perspective'][0]['evaluation']['value'] == -712
    assert actual['child_analyses'][0]['evaluation']['value'] == 712
    assert '分支表不要再次反号' in messages[0]['content']
    assert '子评分来自对手，不得混淆' not in messages[0]['content']
    assert prose_moves.checked_consolidation_payload(actual, PROFILE)['children_with_prose'] == 1


def test_native_cannon_check_requires_second_screen_not_a_rook_block():
    content, target = packet()
    facts = json.loads(rendered(content, target)[-1]['content'])['native_coaching_evidence']
    assert facts['root_in_check'] is True
    assert facts['root_checkers'] == [{'square': 'e5', 'piece': '红炮'}]
    assert facts['candidate_evasions'] == [{'move': 'g9e7', 'checker_mechanisms': [
        {'checker': 'e5', 'mechanism': 'add_cannon_screen'}]}]
    assert facts['lines'][0]['captured_by']['red']['p'] == 1
    assert facts['lines'][0]['material_change']['black']['p'] == -1


def test_horse_check_is_evaded_by_blocking_the_leg():
    fen = 'r2ak1b2/4ar3/b3c1n2/p5p1p/4P4/1CBN5/P3N1P1P/9/2n1A4/R3KABR1 w - - 9 16'
    content, target = packet(fen, ('e3d1', 'c1a0'))
    facts = json.loads(rendered(content, target)[-1]['content'])['native_coaching_evidence']
    assert facts['root_side'] == '红方'
    assert facts['candidate_evasions'][0]['checker_mechanisms'] == [
        {'checker': 'c1', 'mechanism': 'block_horse_leg'}]
    assert facts['lines'][0]['captured_by']['black']['r'] == 1
    assert facts['lines'][0]['material_change']['red']['r'] == -1


def test_both_sides_capture_counts_are_not_confused_with_total_advantage():
    moves = ('e6a6', 'h8h3', 'a6i6', 'h3e3', 'i6e6', 'e3c3', 'e6e5')
    content, target = packet(MATERIAL_FEN, moves)
    evidence = json.loads(rendered(content, target)[-1]['content'])['native_coaching_evidence']
    line = evidence['lines'][0]
    assert len(evidence['lines']) == 1  # Main/selected branch duplicates are counted once.
    assert line['captured_by']['red']['p'] == 3
    assert line['captured_by']['black']['p'] == 1
    assert line['material_change']['red']['p'] == -1
    assert line['material_change']['black']['p'] == -3
    assert evidence['root_material']['red']['p'] == 4
    assert evidence['root_material']['black']['p'] == 4
    assert line['final_material']['red']['p'] == 3
    assert line['final_material']['black']['p'] == 1


@pytest.mark.parametrize('fen,moves,expected_file,expected_rank', [
    (MATERIAL_FEN, ('e6a6',), 9, 6), (CHECK_FEN, ('g9e7',), 5, 7)])
def test_file_numbers_are_relative_to_moving_color_not_destination_rank(
        fen, moves, expected_file, expected_rank):
    content, target = packet(fen, moves)
    line = json.loads(rendered(content, target)[-1]['content'])['native_coaching_evidence']['lines'][0]
    assert line['first_move_destination'] == {
        'file_for_moving_side': expected_file, 'rank_zero_based': expected_rank}


@pytest.mark.parametrize('field', ['root_in_check', 'root_oracle_evaluation', 'lines'])
def test_changed_grounding_cannot_pass_profile_attestation(field):
    content, target = packet()
    actual = json.loads(rendered(content, target)[-1]['content'])
    actual['native_coaching_evidence'][field] = None
    with pytest.raises(ValueError, match='differs from native facts'):
        prose_moves.checked_consolidation_payload(actual, PROFILE)


@pytest.mark.parametrize('bad_value', [True, 1.5, '12'])
def test_untyped_score_cannot_silently_enter_root_table(bad_value):
    content, target = packet()
    target['branches'][0]['evaluation']['value'] = bad_value
    with pytest.raises(ValueError, match='integer evaluations'):
        rendered(content, target)


def test_wrong_packet_root_fails_before_teacher_generation():
    content, target = packet()
    content['root_fen'] = START_FEN
    with pytest.raises(ValueError, match='root and target differ'):
        prose_moves.consolidation_messages(content, CHECK_FEN, target, consolidation_profile=PROFILE)


def test_numeric_child_generation_is_rejected_before_student_work():
    predictor = CapturingPredictor()
    with pytest.raises(ValueError, match='full_explanation'):
        make_search(predictor, 'paper_recursive', consolidation_profile=PROFILE,
                    child_prompt_profile='numeric_six_ply_v1')
    assert not predictor.calls


def test_missing_actual_child_prose_is_not_replaced_by_grounding():
    content, target = packet()
    content['child_analyses'][0]['unverified_explanation'] = ''
    with pytest.raises(prose_moves.MissingChildProseError):
        rendered(content, target)


def test_miner_keeps_the_actual_trace_targets_and_raw_child_strings():
    original = make_search(CapturingPredictor(), 'paper_recursive',
                           consolidation_profile='raw_child_prose_v1').mine(root_record())
    actual = make_search(CapturingPredictor(), 'paper_recursive',
                         consolidation_profile=PROFILE).mine(root_record())
    query, trace, reason = actual
    assert reason == original[2] == 'accepted_for_consolidation'
    assert trace == original[1]
    assert query['target_fields'] == original[0]['target_fields']
    old = json.loads(original[0]['messages'][-1]['content'])
    new = json.loads(query['messages'][-1]['content'])
    assert new['child_analyses'] == old['child_analyses']
    assert new['root_line_facts'] == old['root_line_facts']
    proof = prose_moves.consolidation_queries_evidence([query], PROFILE)
    assert proof['queries'] == 1 and proof['children_with_prose'] > 0
