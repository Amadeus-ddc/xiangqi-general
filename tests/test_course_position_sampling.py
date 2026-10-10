import json
import random
import hashlib

import pytest

from xqgeneral.course_position_sampling import (
    POSITION_PROFILE, MIXES, PositionCatalog, position_budgets, root_targets, weighted_interleave,
)
from xqgeneral.course_sampling import SAMPLING_PROFILE, TRAIN_QUERIES
from xqgeneral.course_tasks import PAPER_TASKS, native_tag, paper_records
from xqgeneral.evidence import atomic_json, history_key, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.paper_curriculum import build, readback
from xqgeneral.recorded_course_pools import terminal_candidates
from xqgeneral.rules import adjudicate, in_check, legal_moves, replay
from xqgeneral.symmetry import mirror_fen
from xqgeneral.tactical_course_pools import source_candidates

CHECK = '4k4/9/9/9/4r4/9/9/9/9/4K4 w - - 0 1'
MATE_LINE = ['e0d0', 'e5d5']


def source_game(fen, moves, split, name):
    return {'initial_fen': fen, 'moves': moves, 'history': replay(fen, moves),
            'native_terminal': adjudicate(fen, moves), 'split': split, 'game_id': name,
            'source_kind': 'constructed_native_probe', 'source': 'constructed_test_probe',
            'headers': {}, 'players': [], 'provenance': 'constructed_native_fixture;not_real_games'}


def fixture(tmp_path):
    values = []
    for split, second_rank in [('train', 0), ('validation', 1), ('test', 2)]:
        for rank in range(9):
            rows = CHECK.split()[0].split('/')
            rows[9 - rank] = 'H3K4' if rank == 0 else 'H3r4' if rank == 5 else 'H8'
            i = 9 - second_rank
            rows[i] = rows[i][:-1] + str(int(rows[i][-1]) - 1) + 'H'
            fen = '/'.join(rows) + ' w - - 0 1'
            game = source_game(fen, MATE_LINE, split, f'{split}-controlled-native-{rank}')
            values += terminal_candidates(game, split)['roots']
    p = tmp_path / 'sources'
    write_jsonl(p / 'roots.jsonl', values)
    atomic_json(p / 'manifest.json', manifest('constructed_native_root_probe', {}, [], [p / 'roots.jsonl'], {}))
    return p / 'roots.jsonl', p / 'manifest.json', values


def small_budgets(n=1):
    return {split: {stage: {task: n for task in tasks} for stage, tasks in PAPER_TASKS.items()}
            for split in ('train', 'validation', 'test')}


def test_author_requested_position_budgets_and_mixtures_match_yaml_counts():
    budgets = position_budgets()
    assert sum(n * TRAIN_QUERIES[t] for t, n in budgets['train']['static_current'].items()) == 12800004
    assert budgets['train']['static_future'] == budgets['train']['static_current']
    assert sum(n * TRAIN_QUERIES[t] for t, n in budgets['train']['dynamic_current'].items()) == 9502860
    assert budgets['train']['dynamic_future'] == budgets['train']['dynamic_current']
    assert {n for tasks in budgets['validation'].values() for n in tasks.values()} == {100}
    assert {n for tasks in budgets['test'].values() for n in tasks.values()} == {1000}
    assert dict(MIXES['mate']) == {'mate': .3, 'near_mate_check': .2, 'in_check': .1, 'stalemate': .05, 'generic': .35}


@pytest.mark.parametrize('kind', ['split', 'stage', 'task', 'zero', 'bool', 'float'])
def test_invalid_position_budget_contracts_fail(kind):
    value = small_budgets()
    if kind == 'split': value.pop('test')
    elif kind == 'stage': value['train'].pop('dynamic_future')
    elif kind == 'task': value['train']['static_current'].pop('piece')
    else: value['train']['static_current']['piece'] = {'zero': 0, 'bool': True, 'float': 1.2}[kind]
    with pytest.raises(ValueError, match='budgets'):
        position_budgets(value)


def test_weighted_interleave_matches_smooth_fractions_then_drops_exhausted_stream():
    def candidates(prefix, count):
        return [{'query_key': f'{prefix}{i}', 'color_key': f'mirror-{prefix}{i}'} for i in range(count)]
    from collections import Counter
    counters = {k: Counter() for k in ('selected', 'duplicates', 'exhausted')}
    rows = list(weighted_interleave([('generic', candidates('g', 8), .2), ('check', candidates('c', 8), .8)], set(), counters))
    assert [s for s, _ in rows[:10]].count('check') == 8
    assert len(rows) == 16 and counters['exhausted'] == {'check': 1, 'generic': 1}
    assert [s for s, _ in rows[-6:]] == ['generic'] * 6


def test_color_orbit_duplicate_is_skipped_without_consuming_another_position():
    from collections import Counter
    counters = {k: Counter() for k in ('selected', 'duplicates', 'exhausted')}
    rows = list(weighted_interleave([('generic', [
        {'query_key': 'a', 'color_key': 'b'}, {'query_key': 'b', 'color_key': 'a'},
        {'query_key': 'c', 'color_key': 'd'}], 1.)], set(), counters))
    assert [r['query_key'] for _, r in rows] == ['a', 'c']
    assert counters['duplicates'] == {'generic': 1}


@pytest.mark.parametrize('task', ['piece', 'checks'])
def test_collected_plain_roots_shuffle_before_rendering_while_mixtures_keep_order(tmp_path, monkeypatch, task):
    catalog = PositionCatalog(tmp_path / 'order.sqlite')
    def candidates(split, stage, label, seed):
        for i in range(100):
            key = f'{split}/{stage}/{label}/{i}'
            yield {'query_key': key, 'color_key': 'mirror/' + key}
    monkeypatch.setattr(catalog, 'candidates', candidates)
    monkeypatch.setattr(catalog, 'selected_root', lambda candidate, stage, task, label: candidate)
    try:
        rows = list(catalog.selections(small_budgets(4), 7, {}))
        stage = 'static_current' if task == 'piece' else 'dynamic_current'
        actual = [root['query_key'].removeprefix(f'test/{stage}/')
                  for root, selected_stage, selected_task in rows if
                  selected_stage == stage and selected_task == task and root['query_key'].startswith('test/')]
        if task == 'piece':
            expected = [f'generic/{i}' for i in range(4)]
            seed = int(hashlib.sha256(b'7/test/static_current/piece/position-order').hexdigest(), 16)
            random.Random(seed).shuffle(expected)
            assert expected != [f'generic/{i}' for i in range(4)]
        else:
            # Three preceding tasks have consumed twelve generic query boards.
            expected = ['have_check/0', 'have_check/1', 'generic/12', 'have_check/2']
        assert actual == expected
    finally:
        catalog.close()


def test_future_difficulty_locks_a_matching_horizon_instead_of_randomly_truncating(tmp_path):
    root = terminal_candidates(source_game(CHECK, MATE_LINE, 'train', 'controlled'), 'train')['roots'][0]
    catalog = PositionCatalog(tmp_path / 'catalog.sqlite')
    try:
        catalog.add(root); catalog.finish()
        for seed in range(10):
            candidate = next(catalog.candidates('train', 'dynamic_future', 'in_check', seed))
            assert candidate['horizon'] == 2
            selected = catalog.selected_root(candidate, 'dynamic_future', 'parries', 'in_check')
            assert selected['future_moves'] == MATE_LINE
            assert in_check(replay(selected['fen'], selected['future_moves'])[-1])
            from xqgeneral.course_sampling import FrequencySampler
            rows = paper_records(selected, random.Random(seed), FrequencySampler(),
                                 only_stage='dynamic_future', only_task='parries')
            assert len(rows) == 1 and native_tag(rows[0]) == '无'
    finally:
        catalog.close()


@pytest.mark.parametrize('kind', ['classes', 'distance', 'witness', 'horizon'])
def test_forged_difficulty_metadata_is_rejected(kind):
    root = terminal_candidates(source_game(CHECK, MATE_LINE, 'train', 'controlled'), 'train')['roots'][0]
    if kind == 'classes': root['paper_source_targets'][0]['classes'] = ['mate']
    elif kind == 'distance': root['paper_source_targets'][0]['plies_before_recorded_mate'] = 1
    elif kind == 'witness': root['paper_recorded_mate_witness'] = MATE_LINE[:1]
    else: root['paper_source_targets'][1]['horizon'] = 2
    with pytest.raises(ValueError, match='witness|targets|distance'):
        root_targets(root)


def test_generated_terminal_alternatives_recover_both_future_classes_from_one_parent(tmp_path):
    fen = '4k4/3R5/5R3/9/9/9/9/9/9/5K3 w - - 0 1'
    game = source_game(fen, [], 'train', 'constructed-position')
    game.update(source_kind='recorded_tactical_position', source_initial_fen=fen, headers={'FEN': fen},
                declared_participants={'Red': None, 'Black': None},
                supplied_history_starts_at_standard_initial_position=False, pre_fragment_game_history_available=False,
                provided_line_is_best_move_label=False, source_comments_or_analysis_branches_used_as_labels=False)
    source = source_candidates(game)
    roots = []
    seen = set()
    for root in source['roots']:
        if root['feature_key'] not in seen: roots.append(root); seen.add(root['feature_key'])
    parent = next(r for r in roots if r['ply'] == 0)
    catalog = PositionCatalog(tmp_path / 'generated.sqlite')
    try:
        for root in roots: catalog.add(root)
        catalog.finish()
        for label in ('mate', 'stalemate'):
            candidate = next(catalog.candidates('train', 'dynamic_future', label, 0))
            selected = catalog.selected_root(candidate, 'dynamic_future', 'mate', label)
            assert selected['feature_key'] == parent['feature_key']
            assert selected['game_id'] == game['game_id']
            final = replay(selected['fen'], selected['future_moves'])[-1]
            assert not legal_moves(final) and in_check(final) is (label == 'mate')
            assert selected['paper_recorded_mate_witness'] == []
            assert selected['pre_fragment_game_history_available'] is False
            if candidate['variant_move']:
                assert selected['position_selected_extension_descendant_feature'] in {r['feature_key'] for r in roots}
    finally:
        catalog.close()


@pytest.mark.parametrize('kind', ['duplicate', 'game_owner', 'future_color', 'illegal_branch'])
def test_catalog_rejects_duplicate_game_future_and_branch_contracts(tmp_path, kind):
    roots, _, values = fixture(tmp_path)
    first = values[0]
    prior = {}
    if kind == 'future_color': prior[position_key(mirror_fen(replay(first['fen'], first['future_moves'])[-1]))] = 'validation'
    catalog = PositionCatalog(tmp_path / 'invalid.sqlite', prior_positions=prior)
    try:
        if kind == 'duplicate':
            catalog.add(first)
            with pytest.raises(ValueError, match='duplicate'): catalog.add(first)
        elif kind == 'game_owner':
            catalog.add(first)
            changed = dict(next(v for v in values if v['split'] == 'validation'), game_id=first['game_id'])
            with pytest.raises(ValueError, match='game'): catalog.add(changed)
        elif kind == 'illegal_branch':
            changed = dict(first, future_branches=[['e0e1']])
            with pytest.raises(ValueError, match='Illegal'): catalog.add(changed)
        else:
            with pytest.raises(ValueError, match='future|color'): catalog.add(first)
    finally:
        catalog.close()


def test_full_mate_witness_beyond_maximum_question_horizon_is_reserved(tmp_path):
    moves = ['e0d0', 'e5e6', 'd0d1', 'e6e7', 'd1d2', 'e7e8',
             'd2d1', 'e8e6', 'd1d0', 'e6e7', 'd0d1', 'e7d7']
    root = terminal_candidates(source_game(CHECK, moves, 'train', 'controlled-long-witness'), 'train')['roots'][0]
    end = replay(root['fen'], moves)[-1]
    assert len(root['future_moves']) == 8 and len(root['paper_recorded_mate_witness']) == 12
    catalog = PositionCatalog(tmp_path / 'witness.sqlite', prior_positions={position_key(mirror_fen(end)): 'test'})
    try:
        with pytest.raises(ValueError, match='witness|color'): catalog.add(root)
    finally:
        catalog.close()


def test_complete_budgeted_generation_has_task_disjoint_query_boards_and_source_readback(tmp_path):
    roots, proof, _ = fixture(tmp_path)
    data = tmp_path / 'questions'
    result = build([roots], [proof], data, sampling_profile=SAMPLING_PROFILE,
                   position_sampling_profile=POSITION_PROFILE, task_position_budgets=small_budgets(2))
    assert result['all_requested_task_root_budgets_filled'] is True
    assert all(result['all_declared_tasks_present_by_split'].values())
    assert len(result['task_position_selection']) == 78
    for split in ('train', 'validation', 'test'):
        owners = {}
        for row in load_jsonl(data / f'{split}.jsonl'):
            assert row['position_selected_stage'] == row['stage']
            assert row['position_selected_task'] == row['task_type']
            assert row['position_selected_horizon'] == len(row['future_moves'])
            for key in (position_key(row['fen']), position_key(mirror_fen(row['fen']))):
                group = row['stage'], key
                assert owners.setdefault(group, row['task_type']) == row['task_type']
            if row['position_source_class'] == 'in_check':
                assert in_check(replay(row['fen'], row['future_moves'])[-1])
    assert not list(data.glob('native-position-catalog-*'))
    checked = readback(data, tmp_path / 'readback')
    assert checked['task_position_selection'] == result['task_position_selection']
    assert checked['questions_regenerated_and_compared'] == result['questions']


def test_finite_source_shortfalls_are_reported_without_silent_replacement(tmp_path):
    roots, proof, _ = fixture(tmp_path)
    result = build([roots], [proof], tmp_path / 'questions', sampling_profile=SAMPLING_PROFILE,
                   position_sampling_profile=POSITION_PROFILE, task_position_budgets=small_budgets(100))
    assert result['all_requested_task_root_budgets_filled'] is False
    assert any(r['shortfall_positions'] > 0 for r in result['task_position_selection'].values())
    assert result['source_streams_are_finite_and_never_cycled'] is True
    assert result['all_declared_tasks_present_by_split']['train'] is False


def test_ordered_parallel_catalog_build_matches_every_question_and_selection_byte(tmp_path):
    roots, proof, _ = fixture(tmp_path)
    a = build([roots], [proof], tmp_path / 'serial', sampling_profile=SAMPLING_PROFILE,
              position_sampling_profile=POSITION_PROFILE, task_position_budgets=small_budgets())
    b = build([roots], [proof], tmp_path / 'parallel', sampling_profile=SAMPLING_PROFILE,
              position_sampling_profile=POSITION_PROFILE, task_position_budgets=small_budgets(), position_workers=2)
    assert a['task_position_selection'] == b['task_position_selection']
    for split in ('train', 'validation', 'test'):
        assert (tmp_path / 'serial' / f'{split}.jsonl').read_bytes() == (tmp_path / 'parallel' / f'{split}.jsonl').read_bytes()
    assert readback(tmp_path / 'parallel', tmp_path / 'parallel-readback')['status'] == 'complete'


@pytest.mark.parametrize('kind', ['profile', 'budget', 'selection', 'horizon'])
def test_readback_rejects_changed_position_profile_policy_and_labels(tmp_path, kind):
    roots, proof, _ = fixture(tmp_path)
    data = tmp_path / 'questions'
    build([roots], [proof], data, sampling_profile=SAMPLING_PROFILE,
          position_sampling_profile=POSITION_PROFILE, task_position_budgets=small_budgets())
    p = data / 'manifest.json';m = json.loads(p.read_text())
    if kind == 'profile': m['config']['position_sampling_profile'] = 'unrecognized'
    elif kind == 'budget': m['config']['task_position_budgets']['train']['static_current']['piece'] = 2
    elif kind == 'selection': next(iter(m['verification']['task_position_selection'].values()))['selected_positions'] = 99
    else:
        path = data / 'train.jsonl';rows = load_jsonl(path)
        next(r for r in rows if r['stage'].endswith('_future'))['position_selected_horizon'] = 0
        write_jsonl(path, rows)
        from xqgeneral.evidence import digest
        m['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(p, m)
    with pytest.raises(ValueError, match='profile|bytes|budgets|shortfalls'):
        readback(data, tmp_path / 'readback')


def test_unrequested_position_settings_and_missing_answer_sampler_are_rejected(tmp_path):
    roots, proof, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='position sampling'):
        build([roots], [proof], tmp_path / 'no-sampler', position_sampling_profile=POSITION_PROFILE)
    with pytest.raises(ValueError, match='position sampling'):
        build([roots], [proof], tmp_path / 'undeclared', task_position_budgets=small_budgets())
