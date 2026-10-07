import pytest
from xqgeneral.human_games import assigned_split, parse_dhtml_game, parse_game, recorded_year, resolve_move
from xqgeneral.recorded_sources import event_group, participant_label, select_diverse_games, source_kind
from xqgeneral.collect_recorded import public_url
from xqgeneral.rules import START_FEN, play


def pgn(body, result='1-0'):
    return ('[Game "Chinese Chess"]\n[Date "2013-01-01"]\n[Red "许银川"]\n'
            '[Black "王天一"]\n[Result "' + result + '"]\n\n' + body).encode()


def test_chinese_big5_game_and_iccs_have_the_same_history_and_identity():
    chinese = pgn('1. 炮二平五 馬８進７ 2. 馬二進三 車９平８ 1-0').decode().replace('许银川', '許銀川').encode('big5')
    game = parse_game(chinese)
    coordinates = parse_game(pgn('1. H2-E2 H9-G7 2. H0-G2 I9-H9 1-0'))
    assert game['moves'] == coordinates['moves'] == ['h2e2', 'h9g7', 'h0g2', 'i9h9']
    assert game['game_id'] == coordinates['game_id']
    assert game['history'] == coordinates['history']
    assert game['source_encoding'] == 'big5'
    assert game['human_move_optimality_proven'] is False


def test_comments_and_variations_cannot_enter_the_recorded_main_line():
    game = parse_game(pgn('1. 炮二平五 {wrong H2-H9} (1. H2-H9 (H9-H0)) 馬８進７ $1\n'
                         '2. 馬二進三 ; ignore I0-I9\n 車９平８ 1-0'))
    assert game['moves'] == ['h2e2', 'h9g7', 'h0g2', 'i9h9']


@pytest.mark.parametrize('body', ['1. H2-G3 1-0', '1. H2-E2 H2-E2 1-0',
                                  '1. 炮二平五 {unclosed 1-0', '1. 炮二平五 (unclosed 1-0',
                                  '1. 炮二平五 0-1', '1. 炮二平五 unknown 1-0'])
def test_illegal_malformed_or_mismatched_records_fail_without_repair(body):
    with pytest.raises(ValueError):
        parse_game(pgn(body))


def test_native_terminal_history_rejects_recorded_later_moves():
    text = ('[Game "Chinese Chess"]\n[Red "许银川"]\n[Black "王天一"]\n'
            '[Result "1/2-1/2"]\n\n'
            '1. B0-C2 B9-C7 2. C2-B0 C7-B9 3. B0-C2 B9-C7 '
            '4. C2-B0 C7-B9 5. B0-C2 B9-C7 6. C2-B0 C7-B9 7. H0-G2 1/2-1/2')
    with pytest.raises(ValueError, match='full-history terminal'):
        parse_game(text.encode())


def test_duplicate_game_headers_do_not_silently_combine_two_games():
    with pytest.raises(ValueError, match='Duplicate PGN headers'):
        parse_game(pgn('1. H2-E2 1-0') + pgn('1. H2-C2 1-0'))


def test_notation_and_split_use_actual_board_and_canonical_game_identity():
    assert resolve_move(START_FEN, '炮二平五') == 'h2e2'
    assert resolve_move(START_FEN, 'C2=5') == 'h2e2'
    with pytest.raises(ValueError):
        resolve_move(play(START_FEN, 'h2e2'), 'h2e2')
    first = parse_game(pgn('1. 炮二平五 馬８進７ 1-0'))
    second = parse_game(pgn('1. H2-E2 H9-G7 1-0'))
    assert assigned_split(first['game_id'], 7) == assigned_split(second['game_id'], 7)


def test_file_number_notation_with_stacked_pieces_requires_a_unique_legal_move():
    # The recorded Lv Qin vs Jiang Chuan game names the file instead of "front cannon".
    fen = '2baka3/9/4b1nc1/p3p3p/3n2c2/8C/P3P3P/2N1B1N1C/9/3AKAB2 w - - 3 20'
    assert resolve_move(fen, '炮一平六') == 'i4d4'
    assert resolve_move(fen, '前炮平二') == 'i4h4'
    assert resolve_move(fen, '后炮平二') == 'i2h2'
    with pytest.raises(ValueError, match='2 legal interpretations'):
        resolve_move(fen, '炮一平二')


def dhtml(length=4, firstnum=0):
    fields = {'binit': '8979695949392919097717866646260600102030405060708012720323436383',
              'firstnum': firstnum, 'length': length, 'type': '全局',
              'event': '2019年测试赛事', 'red': '许银川', 'black': '王天一', 'result': '红胜',
              'movelist': '7747706279678070'}
    text = '[DhtmlXQ]' + ''.join(f'[DhtmlXQ_{k}]{v}[/DhtmlXQ_{k}]' for k, v in fields.items())
    return (text + '[DhtmlXQ_comment1]unverified annotation[/DhtmlXQ_comment1]'
            '[DhtmlXQ_move_0_1_1]invalid branch[/DhtmlXQ_move_0_1_1][/DhtmlXQ]').encode()


def test_dhtml_extracts_main_line_and_preserves_partial_date_precision():
    game = parse_dhtml_game(dhtml())
    assert game['moves'] == ['h2e2', 'h9g7', 'h0g2', 'i9h9']
    assert game['headers']['Date'] == '2019-??-??'
    assert game['date_precision'] == 'event_year_only'
    assert game['source_comments_or_analysis_branches_used'] is False


def test_dhtml_declared_first_player_draw_is_a_source_result_enum():
    game = parse_dhtml_game(dhtml().replace('红胜'.encode(), '先和'.encode()))
    assert game['declared_result'] == '1/2-1/2'


@pytest.mark.parametrize('source', [dhtml(length=63), dhtml(firstnum=20)])
def test_dhtml_partial_analysis_is_not_imported_as_a_complete_game(source):
    with pytest.raises(ValueError):
        parse_dhtml_game(source)


def test_source_classes_keep_computer_and_human_computer_matches_separate():
    assert source_kind('Dataset/對局/大師對局/以棋手分類/a.pgn') == 'recorded_human_match'
    assert source_kind('Dataset/對局/電腦對局/人機賽/a.pgn') == 'recorded_human_computer_match'
    assert source_kind('Dataset/對局/電腦對局/電腦對局競賽/a.pgn') == 'recorded_computer_match'
    assert source_kind('Dataset/殺局_殺法_練習題/a.pgn') is None


def test_participant_labels_and_years_preserve_limits_of_source_metadata():
    assert participant_label('廣東 許銀川　') == '许银川'
    assert participant_label('廣東許銀川') == '许银川'
    assert participant_label('上海胡榮華') == '胡荣华'
    assert participant_label('unknown_prefix许银川') == 'unknown_prefix许银川'
    assert participant_label('FIBChess 3.2') == 'FIBChess 3.2'
    assert recorded_year({'Date': '4/17/1996'}) == 1996
    assert recorded_year({'Date': '2001年9月15日'}) == 2001
    assert recorded_year({'Date': '2013.??.??'}) == 2013
    assert recorded_year({'Date': 'unknown', 'Event': 'published in 2026'}) is None


def test_escaped_pgn_header_quotes_are_facts_rather_than_literal_backslashes():
    source = pgn('1. H2-E2 H9-G7 1-0').replace(b'[Game', b'[Event "Test \\"Cup\\" \\\\ edition"]\n[Game')
    assert parse_game(source)['headers']['Event'] == 'Test "Cup" \\ edition'


def test_diverse_selection_caps_both_players_and_keeps_multiple_eras_and_sources():
    def game(identity, players, year, event='event', kind='recorded_human_match'):
        return {'game_id': identity, 'players': players, 'headers': {'Date': str(year), 'Event': event},
                'source_kind': kind, 'declared_result': '1-0'}
    inputs = [game('a', ['A', 'B'], 2001), game('b', ['C', 'A'], 2002),
              game('c', ['D', 'E'], 1985), game('d', ['F', 'G'], 2012, 'other'),
              game('e', ['engine1', 'engine2'], 2011, 'computer', 'recorded_computer_match')]
    selected, _ = select_diverse_games(inputs, 10, participant_cap=1, event_cap=2, seed=7)
    identities = {g['game_id'] for g in selected}
    assert len(identities & {'a', 'b'}) <= 1
    assert {'c', 'd', 'e'} <= identities
    assert select_diverse_games(list(reversed(inputs)), 10, 1, 2, 7)[0] == selected


@pytest.mark.parametrize('url', ['https://' + 'invalid-user:invalid-password@www.xiangqiqipu.com/Category/View-1.html',
                               'https://www.xiangqiqipu.com/login',
                               'https://example.com/Category/View-1.html'])
def test_public_collector_rejects_credentials_and_nonpublic_source_routes(url):
    with pytest.raises(ValueError):
        public_url(url)


def test_layout_and_round_suffixes_do_not_evade_a_tournament_cap():
    def game(identity, event):
        return {'game_id': identity, 'players': [identity + 'red', identity + 'black'],
                'headers': {'Event': event, 'Date': '1984-01-01'},
                'source_kind': 'recorded_human_match', 'declared_result': '1-0'}
    first = game('a', '1984年全国象棋团体赛 中炮过河车对屏风马')
    second = game('b', '1984年全国象棋团体赛 第三轮')
    assert event_group(first) == event_group(second)
    selected, rejected = select_diverse_games([first, second], 2, 2, 1, 7)
    assert len(selected) == 1 and rejected['event_cap'] == 1
