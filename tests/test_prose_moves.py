import json
import pytest

from xqgeneral.prose_moves import consolidation_messages, notation_lines, validate_prose_moves
from xqgeneral.rules import START_FEN, move_notation


def test_native_notation_matches_real_mined_capture_and_both_sides():
    fen = '3nkab2/4a1c2/9/8N/4P1b2/P4R2P/9/2N1B4/5p3/2rAKAB2 w - - 1 37'
    line = ['e2c0', 'g8g0', 'f0e1', 'd9c7', 'c0e2', 'c7b5']
    target = {'pv':line, 'branches':[{'pv':line}]}
    plies = notation_lines(fen, target)[0]['plies']
    assert [p['wxf'] for p in plies] == ['E5-7', 'C7+8', 'A4+5', 'H4+3', 'E7+5', 'H3+2']
    assert [p['chinese'] for p in plies] == ['相五退七', '炮7进8', '仕四进五',
                                            '马4进3', '相七进五', '马3进2']
    assert [p['side'] for p in plies] == ['红方', '黑方'] * 3


def test_native_notation_handles_front_piece_and_refuses_illegal_move():
    fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/4P4/1NB6/P1P1P3P/1C1A3C1/9/RNBAK4 w - - 0 1'
    assert move_notation(fen, 'b4c6')['chinese'] == '前马进七'
    with pytest.raises(ValueError, match='illegal'):
        move_notation(START_FEN, 'a0a0')


@pytest.mark.parametrize('prose,valid', [
    ('b0c2（马八进七），b9c7（馬２進３）', True),
    ('马八进七', False),
    ('b0c2（马八进九）', False),
    ('b0c2（马八进七），a0a0', False),
    ('b0c2（马八进十）', False),
])
def test_prose_names_must_match_the_adjacent_verified_coordinate(prose, valid):
    line = ['b0c2', 'b9c7']
    assert validate_prose_moves(START_FEN, {'pv':line, 'branches':[{'pv':line}]}, prose)['valid'] == valid


def test_consolidator_receives_native_names_and_omits_unverified_student_prose():
    line = ['b0c2', 'b9c7']
    target = {'pv':line, 'branches':[{'pv':line}]}
    original = {'child_analyses':[{'used_pv':line[1:], 'unverified_explanation':'错误的捉马正文'}],
                'verified_child_analyses':[{'analysis':{'move':line[1], 'explanation':'未经语义核验'}}]}
    messages = consolidation_messages(original, START_FEN, target)
    packet = json.loads(messages[1]['content'])
    assert packet['verified_move_notation'][0]['plies'][0]['chinese'] == '马八进七'
    assert '独立引擎根评分' in packet['score_sources']['target_fields.evaluation']
    assert 'unverified_explanation' not in packet['child_analyses'][0]
    assert 'explanation' not in packet['verified_child_analyses'][0]['analysis']
    assert original['child_analyses'][0]['unverified_explanation'] == '错误的捉马正文'
