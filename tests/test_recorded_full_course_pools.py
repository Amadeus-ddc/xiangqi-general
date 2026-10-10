import hashlib
import json

import pytest

from xqgeneral.course_tasks import validate_context
from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.human_games import assigned_split
from xqgeneral.recorded_full_course_pools import POOL_PROFILE, build, game_roots, readback
from xqgeneral.rules import START_FEN, adjudicate, replay
from xqgeneral.symmetry import mirror_fen

CHECK = '4k4/9/9/9/4r4/9/9/9/9/4K4 w - - 0 1'
MATE_LINE = ['e0d0', 'e5d5']
LONG_MATE = ['e0d0', 'e5e6', 'd0d1', 'e6e7', 'd1d2', 'e7e8',
             'd2d1', 'e8e6', 'd1d0', 'e6e7', 'd0d1', 'e7d7']


def source_game(fen, moves, seed=0):
    identity = hashlib.sha256(json.dumps([fen, moves], separators=(',', ':')).encode()).hexdigest()
    name = 'recorded-' + identity
    return {'game_id': name, 'split': assigned_split(name, seed), 'initial_fen': fen,
            'source_initial_fen': fen, 'moves': moves, 'history': replay(fen, moves),
            'native_terminal': adjudicate(fen, moves), 'source_kind': 'recorded_human_match',
            'source': {'constructed_fixture': True}, 'headers': {'FEN': fen}, 'players': [],
            'provenance': 'constructed_native_fixture;not_real_games'}


def sources(tmp_path, moves=MATE_LINE):
    fens = []
    for rank in range(3):
        rows = CHECK.split()[0].split('/')
        rows[9-rank] = 'H3K4' if rank == 0 else 'H8'
        fens.append('/'.join(rows) + ' w - - 0 1')
    seed = 0
    while len({source_game(f, moves, seed)['split'] for f in fens}) != 3:
        seed += 1
    values = [source_game(f, moves, seed) for f in fens]
    path = tmp_path / 'sources'
    write_jsonl(path / 'games.jsonl', values)
    atomic_json(path / 'manifest.json', manifest('recorded_games_native_import', {'seed': seed}, [],
                                               [path / 'games.jsonl'], {}))
    return path, seed, values


def protected_data(tmp_path, fen, split):
    path = tmp_path / 'prior'
    for i, s in enumerate(('train', 'validation', 'test')):
        rows = CHECK.split()[0].split('/')
        rows[9 - (i + 3)] = '8H'
        current = fen if s == split else '/'.join(rows) + ' w - - 0 1'
        row = {'game_id': 'protected-native-probe-' + s, 'split': s, 'initial_fen': current,
               'fen': current, 'moves': [], 'history': [current], 'feature_key': history_key([current]),
               'future_moves': [], 'future_branches': []}
        write_jsonl(path / f'{s}.jsonl', [row])
    atomic_json(path / 'manifest.json', manifest('prior_native_probe', {}, [],
        [path / f'{s}.jsonl' for s in ('train', 'validation', 'test')], {}))
    return path


def test_every_given_prefix_actual_terminal_and_long_witness_survive_without_sampling_cap():
    game = source_game(CHECK, LONG_MATE)
    result = game_roots(game, game['split'])
    assert [root['ply'] for root, _ in result['roots']] == list(range(13))
    first, footprint = result['roots'][0]
    assert first['future_moves'] == LONG_MATE[:8]
    assert first['paper_recorded_mate_witness'] == LONG_MATE
    assert position_key(mirror_fen(game['history'][-1])) in footprint
    assert result['source_board_terminal_kind'] == 'mate'
    for root, _ in result['roots']:
        assert root['history'] == game['history'][:root['ply']+1]
        assert root['recorded_source_kind'] == game['source_kind']
        assert root['pre_fragment_game_history_available'] is False
        assert root['provided_line_is_best_move_label'] is False
        validate_context(root, check_future=True)


def test_standard_nonterminal_recording_keeps_last_board_and_known_supplied_history():
    game = source_game(START_FEN, ['h2e2', 'h7e7'])
    result = game_roots(game, game['split'])
    assert len(result['roots']) == 3 and result['source_board_terminal_kind'] is None
    last = result['roots'][-1][0]
    assert last['future_moves'] == [] and last['paper_recorded_mate_witness'] == []
    assert last['pre_fragment_game_history_available'] is True
    assert last['fen'] == game['history'][-1]


def test_no_source_prefix_or_question_future_reaches_history_termination_with_legal_moves():
    moves = ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 3
    game = source_game(START_FEN, moves)
    result = game_roots(game, game['split'])
    terminal = result['source_first_native_terminal_ply']
    assert terminal is not None and terminal < len(moves) and result['source_history_ending_with_legal_moves']
    for root, _ in result['roots']:
        assert root['ply'] + len(root['future_moves']) < terminal
        assert root['paper_recorded_mate_witness'] == []
        validate_context(root, check_future=True)


@pytest.mark.parametrize('kind', ['history', 'native_outcome', 'source_fen', 'source_kind', 'negative_ply'])
def test_corrupt_native_source_or_root_contracts_are_rejected(kind):
    game = source_game(CHECK, MATE_LINE)
    if kind == 'history': game['history'][-1] = CHECK
    elif kind == 'native_outcome': game['native_terminal']['winner'] = 'invented'
    elif kind == 'source_fen': game['source_initial_fen'] = START_FEN
    elif kind == 'source_kind': game['source_kind'] = 'recorded_tactical_position'
    with pytest.raises(ValueError, match='native|recorded|source'):
        game_roots(game, game['split'], -1 if kind == 'negative_ply' else 0)


def test_complete_pool_regeneration_and_parallel_source_order_are_byte_identical(tmp_path):
    source, seed, values = sources(tmp_path)
    a = build([source], tmp_path/'serial', split_seed=seed, workers=1)
    b = build([source], tmp_path/'parallel', split_seed=seed, workers=2)
    assert a == b and a['roots_by_split'] == {'train': 3, 'validation': 3, 'test': 3}
    for name in ('roots.jsonl', 'excluded.jsonl'):
        assert (tmp_path/'serial'/name).read_bytes() == (tmp_path/'parallel'/name).read_bytes()
    assert readback(tmp_path/'parallel', tmp_path/'readback', workers=1)['roots_and_exclusions_regenerated']['root'] == 9
    roots = load_jsonl(tmp_path/'serial/roots.jsonl')
    assert all(root['paper_source_pool_profile'] == POOL_PROFILE for root in roots)
    assert {root['recorded_source_game_id'] for root in roots} == {game['game_id'] for game in values}


def test_reservation_includes_whole_witness_even_beyond_the_eight_ply_question_horizon(tmp_path):
    source, seed, games = sources(tmp_path, LONG_MATE)
    game = games[0]
    other = 'test' if game['split'] != 'test' else 'validation'
    prior = protected_data(tmp_path, mirror_fen(game['history'][-1]), other)
    result = build([source], tmp_path/'pool', reference_data=[prior], split_seed=seed, workers=1)
    roots = load_jsonl(tmp_path/'pool/roots.jsonl')
    assert not any(root['game_id'] == game['game_id'] and root['ply'] == 0 for root in roots)
    assert result['excluded_candidates_by_reason']['root_future_witness_or_color_overlaps_another_split'] > 0


def test_shared_opening_geometry_is_excluded_without_moving_source_games(tmp_path):
    source, seed, games = sources(tmp_path)
    root = game_roots(games[0], games[0]['split'])['roots'][0][0]
    other = next(game['split'] for game in games if game['split'] != root['split'])
    prior = protected_data(tmp_path, root['fen'], other)
    build([source], tmp_path/'pool', reference_data=[prior], split_seed=seed, workers=1)
    kept = load_jsonl(tmp_path/'pool/roots.jsonl')
    assert not any(r['feature_key'] == root['feature_key'] for r in kept)
    owners = {game['game_id']:game['split'] for game in games}
    assert all(row['split'] == owners[row['game_id']] for row in kept)


def test_previously_assigned_course_owner_survives_different_archive_defaults(tmp_path):
    source, seed, games = sources(tmp_path)
    previous = {'train': 'validation', 'validation': 'test', 'test': 'train'}
    owners = {game['game_id']: previous[game['split']] for game in games}
    path = tmp_path / 'previous-owners'
    write_jsonl(path / 'roots.jsonl', [game_roots(game, owners[game['game_id']])['roots'][0][0]
                                     for game in games])
    atomic_json(path / 'manifest.json', manifest('prior_recorded_root_owners', {}, [],
                                                [path / 'roots.jsonl'], {}))
    result = build([source], tmp_path / 'pool', owners=path / 'roots.jsonl',
                   owner_manifest=path / 'manifest.json', split_seed=seed, workers=1)
    assert result['roots_by_split'] == {'train': 3, 'validation': 3, 'test': 3}
    original = {game['game_id']: game['split'] for game in games}
    for root in load_jsonl(tmp_path / 'pool/roots.jsonl'):
        assert root['split'] == owners[root['game_id']]
        assert root['source_archive_split'] == original[root['game_id']]
    readback(tmp_path / 'pool', tmp_path / 'readback', workers=1)


@pytest.mark.parametrize('same_split', [True, False])
def test_distinct_source_games_with_shared_prefix_are_deduplicated_or_isolated(tmp_path, same_split):
    source, _, games = sources(tmp_path)
    fen = games[0]['initial_fen']
    seed = 0
    while True:
        values = [source_game(game['initial_fen'], MATE_LINE, seed) for game in games]
        extra = source_game(fen, ['e0d0', 'e5e6'], seed)
        if (len({game['split'] for game in values}) == 3 and
                (extra['split'] == values[0]['split']) is same_split):
            break
        seed += 1
    write_jsonl(source / 'games.jsonl', [*values, extra])
    atomic_json(source / 'manifest.json', manifest('recorded_games_native_import', {'seed': seed}, [],
                                                  [source / 'games.jsonl'], {}))
    result = build([source], tmp_path / 'pool', split_seed=seed, workers=1)
    roots = load_jsonl(tmp_path / 'pool/roots.jsonl')
    if same_split:
        assert len(roots) == 10
        assert result['excluded_candidates_by_reason'] == {'duplicate_full_history_root': 2}
    else:
        shared = history_key([fen])
        assert not any(root['feature_key'] == shared for root in roots)
        assert result['excluded_candidates_by_reason']['root_future_witness_or_color_overlaps_another_split'] > 0
    owners = {game['game_id']: game['split'] for game in [*values, extra]}
    assert all(root['split'] == owners[root['game_id']] for root in roots)
    readback(tmp_path / 'pool', tmp_path / 'readback', workers=1)


def test_tactical_given_position_can_reserve_geometry_without_becoming_a_full_match(tmp_path):
    source, seed, games = sources(tmp_path)
    isolated = source_game(CHECK, [], seed)
    isolated.update(source_kind='recorded_tactical_position', declared_participants={'Red': None, 'Black': None},
                    supplied_history_starts_at_standard_initial_position=False,
                    pre_fragment_game_history_available=False, provided_line_is_best_move_label=False,
                    source_comments_or_analysis_branches_used_as_labels=False)
    path = tmp_path / 'isolated-source'
    write_jsonl(path / 'games.jsonl', [isolated])
    atomic_json(path / 'manifest.json', manifest('isolated_tactical_positions_native_import', {'seed': seed}, [],
                                                [path / 'games.jsonl'], {}))
    result = build([source, path], tmp_path / 'pool', split_seed=seed, workers=1)
    assert sum(result['source_games_by_split_kind'].values()) == len(games)
    assert all(root['game_id'] != isolated['game_id'] for root in load_jsonl(tmp_path / 'pool/roots.jsonl'))
    with pytest.raises(ValueError, match='every preassigned split'):
        build([path], tmp_path / 'no-full-matches', split_seed=seed, workers=1)


def test_complete_source_roots_feed_all_twenty_six_tasks_and_native_corpus_readback(tmp_path):
    from xqgeneral.course_position_sampling import POSITION_PROFILE
    from xqgeneral.course_sampling import SAMPLING_PROFILE
    from xqgeneral.course_tasks import PAPER_TASKS
    from xqgeneral.paper_curriculum import build as questions, readback as question_readback

    fens = []
    for second_rank in range(3):
        for rank in range(9):
            rows = CHECK.split()[0].split('/')
            rows[9-rank] = 'H3K4' if rank == 0 else 'H3r4' if rank == 5 else 'H8'
            i = 9-second_rank
            rows[i] = rows[i][:-1] + str(int(rows[i][-1])-1) + 'H'
            fens.append('/'.join(rows) + ' w - - 0 1')
    # A fixed source-ID seed gives 11/8/8 games while retaining archive split rules.
    seed = 36207
    values = [source_game(fen, MATE_LINE, seed) for fen in fens]
    assert min(sum(game['split'] == split for game in values)
               for split in ('train', 'validation', 'test')) >= 7
    source = tmp_path / 'full-sources'
    write_jsonl(source / 'games.jsonl', values)
    atomic_json(source / 'manifest.json', manifest('recorded_games_native_import', {'seed': seed}, [],
                                                  [source / 'games.jsonl'], {}))
    pool = tmp_path / 'full-pool'
    build([source], pool, split_seed=seed, workers=1)
    budgets = {split: {stage: {task: 1 for task in tasks} for stage, tasks in PAPER_TASKS.items()}
               for split in ('train', 'validation', 'test')}
    result = questions([pool / 'roots.jsonl'], [pool / 'manifest.json'], tmp_path / 'questions',
                       sampling_profile=SAMPLING_PROFILE, position_sampling_profile=POSITION_PROFILE,
                       task_position_budgets=budgets, position_workers=1)
    assert all(result['all_declared_tasks_present_by_split'].values())
    assert result['all_requested_task_root_budgets_filled'] is True
    proof = question_readback(tmp_path / 'questions', tmp_path / 'question-readback')
    assert proof['status'] == 'complete'


@pytest.mark.parametrize('kind', ['source', 'root', 'provenance', 'verification'])
def test_complete_readback_rejects_changed_sources_roots_provenance_and_claims(tmp_path, kind):
    source, seed, _ = sources(tmp_path)
    data=tmp_path/'pool';build([source], data, split_seed=seed, workers=1)
    p=data/'manifest.json';proof=json.loads(p.read_text())
    if kind == 'source':
        with (source/'games.jsonl').open('a') as out:out.write('\n')
    elif kind in ('root', 'provenance'):
        rows=load_jsonl(data/'roots.jsonl')
        if kind == 'root': rows[0]['future_moves']=[]
        else: rows[0]['pre_fragment_game_history_available']=True
        write_jsonl(data/'roots.jsonl',rows)
        proof['outputs'][str(data/'roots.jsonl')]={'sha256':digest(data/'roots.jsonl'),'bytes':(data/'roots.jsonl').stat().st_size}
    else:
        proof['verification']['all_eligible_source_prefixes_examined_without_per_game_sampling_cap']=False
    atomic_json(p,proof)
    with pytest.raises(ValueError,match='changed|differ'):
        readback(data,tmp_path/'readback',workers=1)


@pytest.mark.parametrize('kwargs', [{'workers':0}, {'workers':True}, {'min_ply':-1}, {'min_ply':False}, {'owners':'unbound'}])
def test_undeclared_or_invalid_generation_settings_fail_before_producing_data(tmp_path, kwargs):
    source,seed,_=sources(tmp_path)
    with pytest.raises(ValueError,match='full-recorded'):
        build([source],tmp_path/'pool',split_seed=seed,**kwargs)
    assert not (tmp_path/'pool').exists()
