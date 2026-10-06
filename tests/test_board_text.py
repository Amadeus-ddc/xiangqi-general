from xqgeneral.rules import START_FEN
from xqgeneral.training import messages


def test_explicit_board_has_all_squares_and_does_not_include_any_target_answer():
    record = {'fen': START_FEN, 'question': '建议走哪一步？', 'answer': 'secret-supervision'}
    plain = messages(record)[1]['content']
    dictionary = messages(record, 'bridge', 'dictionary')[1]['content']
    assert 'a0:红车' in dictionary and 'e9:黑将' in dictionary and 'e4:空' in dictionary
    assert dictionary.count(':') == 90
    assert 'secret-supervision' not in dictionary and '棋盘各格' not in plain
    assert messages(record, 'text_lora', 'dictionary') == messages(record, 'bridge', 'dictionary')
