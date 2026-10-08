import json
from pathlib import Path
import sys
import pytest

from xqgeneral.collect_search import consolidated_label
from xqgeneral.curriculum_data import verify_splits
from xqgeneral.explanations import continuation_positions, move_facts
from xqgeneral.rules import START_FEN, play, replay
from xqgeneral.search_distillation import reserved_positions
from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, manifest, position_key, write_jsonl


TEACHER = {'repository':'Qwen/Qwen3.8-27B', 'revision':'pinned', 'quantization':'none',
           'inference_dtype':'bfloat16'}


def query():
    primary, alternate = ['b0c2', 'b9c7'], ['h0g2', 'h9g7']
    score = {'type':'cp', 'value':0, 'perspective':'side_to_move'}
    return {'id':'q-search', 'source_root_id':'q', 'depth':0,
            'record':{'fen':START_FEN, 'initial_fen':START_FEN, 'moves':[], 'split':'train',
                      'game_id':'train-game', 'provenance':'controlled', 'feature_key':'root'},
            'target_fields':{'move':primary[0], 'pv':primary, 'candidates':[primary[0], alternate[0]],
                'branches':[{'move':line[0], 'pv':line, 'evaluation':score} for line in [primary,alternate]],
                'facts':move_facts(START_FEN, primary[0]), 'evaluation':score}}


def response():
    return {'id':'q-search', 'hit_generation_limit':False,
            'teacher':{**TEACHER, 'parameter_elements_by_dtype':{'torch.bfloat16':27}},
            'text':json.dumps({'explanation':'只按给定事实解释出马和对方应对，分清根行棋方和对手的评分视角；'
                '两种变化都须核对吃子和将军。评分不能证明强制获胜，要根据后续局面比较子力活动及风险，'
                '避免把未给出的变化或进攻效果当成已经发生的结果。'}, ensure_ascii=False)}


def test_collected_search_preserves_every_branch_for_split_verification(tmp_path):
    q = query()
    row = consolidated_label(q, response(), TEACHER)
    assert row['future_moves'] == ['b0c2', 'b9c7']
    assert row['future_branches'] == [['b0c2', 'b9c7'], ['h0g2', 'h9g7']]
    forbidden_fen = replay(START_FEN, ['h0g2', 'h9g7'])[-1]
    assert position_key(forbidden_fen) in continuation_positions(row, json.loads(row['answer']))
    heldout = {'fen':forbidden_fen, 'split':'validation', 'game_id':'heldout', 'future_moves':[]}
    # Identical roots and primary PVs must not cause a distinct branch to be skipped.
    plain = dict(row, future_branches=[])
    with pytest.raises(ValueError, match='leakage'):
        verify_splits([plain, row, heldout])
    source = dict(row, split='validation', future_moves=[], future_branches=[])
    write_jsonl(tmp_path/'validation.jsonl', [source]);write_jsonl(tmp_path/'test.jsonl', [])
    assert position_key(forbidden_fen) in reserved_positions(tmp_path)


def test_search_collection_refuses_history_terminal_continuation_and_missing_branches():
    q = query()
    q['target_fields'].pop('branches')
    with pytest.raises(ValueError, match='missing_branches'):
        consolidated_label(q, response(), TEACHER)
    q = query()
    moves = (['b0c2','b9c7','c2b0','c7b9'] * 2)[:-1]
    q['record'].update(moves=moves, fen=replay(START_FEN,moves)[-1])
    target = q['target_fields'];line=['c7b9','b0c2']
    target.update(move=line[0], pv=line, candidates=[line[0]], facts=move_facts(q['record']['fen'],line[0]))
    target['branches']=[{'move':line[0], 'pv':line, 'evaluation':target['evaluation']}]
    with pytest.raises(ValueError, match='terminal history'):
        consolidated_label(q, response(), TEACHER)


def mined_capture_query():
    fen = '3nkab2/4a1c2/9/8N/4P1b2/P4R2P/9/2N1B4/5p3/2rAKAB2 w - - 1 37'
    lines = [['e2c0', 'g8g0', 'f0e1', 'd9c7', 'c0e2', 'c7b5'],
             ['i6g5', 'c0c2', 'f4c4', 'c2b2', 'c4c0', 'f1g1'],
             ['f0e1', 'c0c2', 'f4c4', 'c2b2', 'c4b4', 'g8i8']]
    q = query()
    q['record'].update(fen=fen, initial_fen=fen, moves=[])
    q['target_fields'].update(move=lines[0][0], pv=lines[0],
        candidates=[line[0] for line in lines], facts=move_facts(fen, lines[0][0]),
        branches=[{'move':line[0], 'pv':line,
                   'evaluation':{'type':'cp', 'value':score, 'perspective':'side_to_move'}}
                  for line, score in zip(lines, [623, -199, -199])])
    q['target_fields']['evaluation']['value'] = 677
    return q


@pytest.mark.parametrize('prose', [
    '红方推荐相七进九吃车，评分677，确立优势。黑方应以炮8平1将军，红仕五进六挡将。'
    '随后黑马9进7，红相九进七回防，黑马7进5。此变化中红方虽失一相，但成功消除黑车威胁，'
    '且子力位置更协调。备选马八进七吃象评分-199，黑车2进4捉马后红方陷入被动。',
    '推荐e2c0（相七进五），之后按给定变化分析对方的炮将军和红方的应对。两条备选的评分'
    '都比推荐着法低，具体吃子和将军须以每步棋盘事实为准。该有限变化与评分不能证明强制'
    '获胜，也不能把没有给出的后续攻击当作已经发生的结果。',
    '推荐e2c0吃车，再考虑f4f5加强进攻。两条备选的评分都比推荐着法低，具体吃子和将军'
    '须以每步棋盘事实为准。该有限变化与评分不能证明强制获胜，也不能把没有给出的后续'
    '攻击当作已经发生的结果，讲解必须保持与实际提供的变化一致。',
])
def test_search_collection_rejects_false_or_ungrounded_prose_moves(prose):
    raw = response()
    raw['text'] = json.dumps({'explanation':prose}, ensure_ascii=False)
    with pytest.raises(ValueError, match='prose[ _]move'):
        consolidated_label(mined_capture_query(), raw, TEACHER)


def test_search_collection_accepts_paired_native_notation_without_changing_prose():
    prose = ('推荐e2c0（相五退七）吃掉黑车。主线g8g0（炮7进8）吃红相并将军，'
             '红方f0e1（仕四进五）移开炮与帅之间的炮架，解除将军。备选i6g5吃象，'
             '黑方c0c2吃马，评分较低。根评分677是红方视角；给定变化不能证明强制获胜。')
    raw = response()
    raw['text'] = json.dumps({'explanation':prose}, ensure_ascii=False)
    row = consolidated_label(mined_capture_query(), raw, TEACHER)
    assert json.loads(row['answer'])['explanation'] == prose


def recorded_collection(tmp_path, *, reserved_alternate=False, descendant=False):
    from test_recorded_search_inputs import fixture
    from xqgeneral.recorded_search_inputs import prepare
    from xqgeneral.symmetry import mirror_fen
    args, _ = fixture(tmp_path)
    games = load_jsonl(args['games'][0] / 'games.jsonl')
    for split in ('validation', 'test'):
        game = next(g for g in games if g['split'] == split)
        row = {'id': '保留-' + split, 'game_id': game['game_id'], 'split': split,
               'initial_fen': game['initial_fen'], 'moves': game['moves'], 'history': game['history'],
               'fen': game['history'][-1], 'feature_key': history_key(game['history']),
               'stage': 'explanation', 'future_moves': [], 'future_branches': []}
        from xqgeneral.rules import legal_moves
        move = sorted(legal_moves(row['fen']))[0]
        score = {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'}
        value = {'move': move, 'pv': [move], 'candidates': [move],
                 'branches': [{'move': move, 'pv': [move], 'evaluation': score}],
                 'facts': move_facts(row['fen'], move), 'evaluation': score,
                 'explanation': '只用于检查保留集格式的受控标签，不代表真实教师标注。'}
        row['answer'] = json.dumps(value, ensure_ascii=False)
        path = args['heldout_data'] / f'{split}.jsonl'
        path.write_text(json.dumps(row, ensure_ascii=True, separators=(',', ':')) + '  \n')
    label_paths = [args['heldout_data'] / f'{s}.jsonl' for s in ('train', 'validation', 'test')]
    atomic_json(args['heldout_data'] / 'manifest.json', manifest('controlled_labels', {}, [], label_paths))
    first = tmp_path / 'first-pool'
    prepare(**args, output=first, min_ply=1, per_game=3)
    original = load_jsonl(first / 'roots.jsonl')[0]
    row = dict(original)
    if descendant:
        move = row['future_moves'][0]
        history = replay(row['initial_fen'], row['moves'] + [move])
        row.update(id=row['id'] + '-' + move, moves=row['moves'] + [move], history=history,
                   fen=history[-1], feature_key=history_key(history), future_moves=[])
    reserved = set(json.loads((first / 'heldout-positions.json').read_text()))
    recorded_future = {position_key(fen) for fen in replay(original['fen'], original['future_moves'])}
    from xqgeneral.rules import legal_moves
    available = [move for move in sorted(legal_moves(row['fen']))
                 if position_key(play(row['fen'], move)) not in reserved | recorded_future]
    assert len(available) >= 2
    lines = [[move] for move in available[:2]]
    if reserved_alternate:
        protected = position_key(mirror_fen(play(row['fen'], lines[1][0])))
        path = args['footprints'].parent / 'footprints.json'
        footprints = json.loads(path.read_text())
        footprints['combined_positions']['validation'].append(protected)
        atomic_json(path, footprints)
        proof = json.loads(args['footprints'].read_text())
        proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        atomic_json(args['footprints'], proof)
        pool = tmp_path / 'isolated-pool'
        prepare(**args, output=pool, min_ply=1, per_game=3)
        assert load_jsonl(pool / 'roots.jsonl')[0]['id'] == original['id']
    else:
        pool = first
    score = {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'}
    q = {'id': row['id'] + '-search', 'source_root_id': original['id'],
         'depth': int(descendant), 'record': row,
         'target_fields': {'move': lines[0][0], 'pv': lines[0],
            'candidates': [line[0] for line in lines],
            'branches': [{'move': line[0], 'pv': line, 'evaluation': score} for line in lines],
            'facts': move_facts(row['fen'], lines[0][0]), 'evaluation': score}}
    response_row = response()
    response_row['id'] = q['id']
    miner = tmp_path / 'miner'; miner.mkdir()
    write_jsonl(miner / 'queries.jsonl', [q])
    write_jsonl(tmp_path / 'responses.jsonl', [response_row])
    atomic_json(tmp_path / 'teachers.json', {'search_consolidator': TEACHER})
    counts = json.loads((pool / 'counts.json').read_text())
    config = {'recorded_inputs': str(pool), 'data': str(args['data']), 'seed': 20261013,
              'limit': 1, 'max_depth': 5}
    inputs = [args['data'] / f'{split}.jsonl' for split in ('train', 'validation', 'test')]
    inputs += [pool / name for name in ('manifest.json', 'roots.jsonl', 'counts.json', 'heldout-positions.json')]
    proof = {name: True for name in ['unused_recorded_training_inputs', 'training_roots_only',
        'heldout_positions_excluded', 'all_child_branch_positions_isolated',
        'raw_child_reachable_prefixes_isolated', 'all_inferred_target_lines_history_validated',
        'strict_pv_improvement_required', 'all_successful_oracle_queries_preserved']}
    proof.update(accepted_for_consolidation=1,
        additional_reserved_canonical_and_explanation_positions=counts['reserved_root_and_future_positions'])
    atomic_json(miner / 'manifest.json', manifest('search_distillation_mining', config, inputs,
        [miner / 'queries.jsonl'], proof))
    return args, pool, miner, q


def collect_args(tmp_path, args, miner, **overrides):
    values = {'queries': miner / 'queries.jsonl', 'responses': tmp_path / 'responses.jsonl',
              'config': tmp_path / 'teachers.json', 'heldout-data': args['data'],
              'validation-data': args['heldout_data'], 'output': tmp_path / 'accepted'}
    values.update(overrides)
    return ['collect_search', *[part for key, value in values.items() for part in ('--' + key, str(value))]]


@pytest.mark.parametrize('descendant', [False, True])
def test_recorded_collection_auto_binds_pool_and_preserves_valid_root_or_descendant(tmp_path, monkeypatch, descendant):
    from xqgeneral.collect_search import main
    args, pool, miner, q = recorded_collection(tmp_path, descendant=descendant)
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner))
    main()
    output = tmp_path / 'accepted'
    rows = load_jsonl(output / 'train.jsonl')
    assert len(rows) == 1 and rows[0]['feature_key'] == q['record']['feature_key']
    assert rows[0]['future_branches'] == [b['pv'] for b in q['target_fields']['branches']]
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['verification']['recorded_inputs'] == str(pool)
    assert proof['verification']['recorded_mining_isolation_rechecked_during_collection'] is True
    assert proof['verification']['student_search_or_oracle_comparisons_recomputed'] is False
    assert str(pool / 'heldout-positions.json') in proof['inputs']
    assert str(args['data'] / 'train.jsonl') in proof['inputs']
    for split in ('validation', 'test'):
        assert (output / f'{split}.jsonl').read_bytes() == (args['heldout_data'] / f'{split}.jsonl').read_bytes()


def test_recorded_collection_rejects_alternate_branch_protected_only_by_color_reservation(tmp_path, monkeypatch):
    from xqgeneral.collect_search import main
    args, pool, miner, q = recorded_collection(tmp_path, reserved_alternate=True)
    target_positions = continuation_positions(q['record'], q['target_fields'])
    assert not target_positions & (reserved_positions(args['data']) | reserved_positions(args['heldout_data']))
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner))
    with pytest.raises(RuntimeError, match='empty dataset'):
        main()
    rejected = json.loads((tmp_path / 'accepted/rejected.json').read_text())
    assert rejected == [{'id': q['id'], 'reason': 'Consolidated root or branch overlaps a heldout position'}]
    assert not (tmp_path / 'accepted/manifest.json').exists()


@pytest.mark.parametrize('change', ['coverage', 'isolation', 'pool_binding', 'seed', 'wrong_base', 'lost_pool'])
def test_recorded_collection_rejects_incomplete_or_mismatched_miner_contract(tmp_path, monkeypatch, change):
    from xqgeneral.collect_search import main
    args, pool, miner, q = recorded_collection(tmp_path)
    p = miner / 'manifest.json'; proof = json.loads(p.read_text())
    if change == 'coverage':
        proof['verification']['accepted_for_consolidation'] = 2
    elif change == 'isolation':
        proof['verification']['raw_child_reachable_prefixes_isolated'] = False
    elif change == 'pool_binding':
        proof['inputs'].pop(str(pool / 'heldout-positions.json'))
    elif change == 'seed':
        proof['config']['seed'] += 1
    elif change == 'wrong_base':
        proof['config']['data'] = str(args['heldout_data'])
    else:
        proof['config']['recorded_inputs'] = None
    atomic_json(p, proof)
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner))
    with pytest.raises(ValueError):
        main()
    assert not (tmp_path / 'accepted').exists()


@pytest.mark.parametrize('change', ['query_bytes', 'nonselected_root', 'history_key'])
def test_recorded_collection_rejects_query_tampering_or_false_training_ownership(tmp_path, monkeypatch, change):
    from xqgeneral.collect_search import main
    args, pool, miner, q = recorded_collection(tmp_path)
    if change == 'history_key':
        q['record']['feature_key'] = 'wrong-context'
    else:
        q['source_root_id'] = 'unselected-root'
    write_jsonl(miner / 'queries.jsonl', [q])
    if change != 'query_bytes':
        p = miner / 'manifest.json';proof = json.loads(p.read_text())
        proof['outputs'][str(miner / 'queries.jsonl')] = {
            'sha256': digest(miner / 'queries.jsonl'), 'bytes': (miner / 'queries.jsonl').stat().st_size}
        atomic_json(p, proof)
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner))
    with pytest.raises(ValueError):
        main()
    assert not (tmp_path / 'accepted').exists()


def test_recorded_collection_requires_miner_proof_when_pool_explicitly_requested(tmp_path, monkeypatch):
    from xqgeneral.collect_search import main
    args, pool, miner, q = recorded_collection(tmp_path)
    (miner / 'manifest.json').unlink()
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner, **{'recorded-inputs': pool}))
    with pytest.raises(ValueError, match='completed mining manifest'):
        main()
    assert not (tmp_path / 'accepted').exists()


@pytest.mark.parametrize('unrelated_manifest', [False, True])
def test_recorded_queries_cannot_silently_fall_back_when_miner_metadata_is_missing(tmp_path, monkeypatch, unrelated_manifest):
    from xqgeneral.collect_search import main
    args, pool, miner, q = recorded_collection(tmp_path)
    p = miner / 'manifest.json'
    if unrelated_manifest:
        atomic_json(p, {'kind': 'unrelated_fixture', 'status': 'complete', 'config': {}})
    else:
        p.unlink()
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner))
    with pytest.raises(ValueError, match='mining|Recorded'):
        main()
    assert not (tmp_path / 'accepted').exists()


def test_recorded_collection_accepts_byte_identical_heldout_copy_and_rejects_changed_labels(tmp_path, monkeypatch):
    from xqgeneral.collect_search import main
    args, pool, miner, q = recorded_collection(tmp_path)
    copied = tmp_path / 'copied-labels'; copied.mkdir()
    for split in ('validation', 'test'):
        (copied / f'{split}.jsonl').write_bytes((args['heldout_data'] / f'{split}.jsonl').read_bytes())
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner, **{'validation-data': copied}))
    main()
    (copied / 'test.jsonl').write_text('\n')
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner,
        **{'validation-data': copied, 'output': tmp_path / 'rejected-copy'}))
    with pytest.raises(ValueError, match='changed'):
        main()
    assert not (tmp_path / 'rejected-copy').exists()
