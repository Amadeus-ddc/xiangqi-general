import copy
import json

import pytest

from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, manifest, write_jsonl
from xqgeneral.explanations import line_facts, move_facts, parse_explanation
from xqgeneral.recorded_coach_prose import (collect_reviewed, prepare_repairs, prepare_review, resolve_review,
                                          resolved_packets, review_packet)
from xqgeneral.revise_prose import annotation_hash, TEACHER
from xqgeneral.rules import START_FEN, legal_moves, piece_map, piece_name, play, side


PROSE = '这段受控测试讲解用于验证数据收集与复核合同，不代表真实教师质量。推荐走法及示例变化的棋子、吃子与将军事实必须从原生规则读取，评分始终按照根局面的行棋方理解，有限搜索并不证明唯一最优或必胜。'


def make_query(index, split, first):
    fen = play(START_FEN, first)
    moves = legal_moves(fen)[:2]
    row = {'id': f'fixture-{index}', 'game_id': f'game-{index}', 'split': split,
           'initial_fen': START_FEN, 'moves': [first], 'history': [START_FEN, fen], 'fen': fen,
           'feature_key': history_key([START_FEN, fen]), 'future_moves': [moves[1]],
           'future_branches': [], 'provenance': 'controlled_legal_fixture',
           'recorded_source_headers': {'Red': 'hidden identity', 'Date': '2021-01-01'}}
    candidates = [{'move': m, 'pv': [m], 'score_type': 'cp', 'score': -index,
                   'perspective': 'side_to_move'} for m in moves]
    facts = {'side_to_move': side(fen), 'recommended': moves[0],
             'board': {s: piece_name(p) for s, p in sorted(piece_map(fen).items())},
             'move_facts': move_facts(fen, moves[0]),
             'branches': [dict(c, line_facts=line_facts(fen, c['pv'])) for c in candidates]}
    return {'id': row['id'] + '-explanation', 'record': row, 'feature_key': row['feature_key'],
            'oracle': {'best_move': moves[0], 'candidates': candidates, 'requested_nodes': 100000},
            'verified_facts': facts}


def source_fixture(tmp_path):
    root = tmp_path / 'source'
    queries = [make_query(i, s, m) for i, (s, m) in enumerate(
        [('train', 'b0c2'), ('validation', 'h0g2'), ('test', 'b2e2')])]
    outputs = []
    for split in ('train', 'validation', 'test'):
        path = root / f'{split}.queries.jsonl'
        write_jsonl(path, [q for q in queries if q['record']['split'] == split]); outputs.append(path)
    agents = []
    for index, q in enumerate(sorted(queries, key=lambda q: (q['record']['split'], q['id']))):
        path = root / 'shards' / f'queries-{index}.jsonl'
        write_jsonl(path, [q]); outputs.append(path)
        annotations = root / 'shards' / f'annotations-{index}.jsonl'
        write_jsonl(annotations, [{'id': q['id'], 'explanation': PROSE, **TEACHER}])
        progress = root / 'shards' / f'author-{index}-progress.json'
        atomic_json(progress, {'status': 'complete', 'expected': 1, 'completed': 1,
                              'input_sha256': digest(path), 'output_sha256': digest(annotations)})
        agents.append({'task_name': f'/root/fixture_author_{index}', 'input': str(path),
                       'input_sha256': digest(path), 'expected': 1,
                       'annotations_output': str(annotations), 'progress': str(progress)})
    counts = {'train': 1, 'validation': 1, 'test': 1}
    source = root / 'queries.manifest.json'
    atomic_json(source, manifest('controlled_original_query_fixture', {}, [], outputs,
                                {'selected_original_teacher_queries': 3, 'by_split': counts}))
    isolation = tmp_path / 'isolation.json'
    atomic_json(isolation, manifest('recorded_coach_selected_query_isolation_readback', {},
        [source, *outputs], [], {'all_queries_and_annotation_shards_match_original_candidates_exactly': True,
        'selected_game_overlap': {'train/test': 0, 'train/validation': 0, 'validation/test': 0},
        'selected_forecast_overlap': {'train/test': 0, 'train/validation': 0, 'validation/test': 0},
        'actual_original_teacher_queries': 3, 'by_split': counts}))
    author_plan = tmp_path / 'authors.json'
    atomic_json(author_plan, {**TEACHER, 'source_queries': str(root),
        'qualified_readback_manifest_sha256': digest(isolation),
        'expected_original_annotations': 3, 'by_split': counts, 'agents': agents})
    return root, isolation, author_plan


def decisions(path, packets_path, verdicts=None, actor='/root/fixture_reviewer'):
    packets = load_jsonl(packets_path)
    result = [{'id': p['id'], 'reviewed_annotation_sha256': p['reviewed_annotation_sha256'],
               'verdict': (verdicts or {}).get(p['id'], 'accept'), 'reason': '受控复核合同测试的逐条判断',
               'issues': ['受控拒收'] if (verdicts or {}).get(p['id']) == 'reject' else []}
              for p in packets]
    atomic_json(path, {'reviewer_model': 'gpt-6-astra', 'reasoning_effort': 'low',
                      'backend': 'codex_subagent', 'human_rating': False, 'source_labels_modified': False,
                      'reviewer_agent': actor, 'input_packet_sha256': digest(packets_path), 'results': result})
    return path


def prepare_with_decisions(tmp_path, shard=None):
    source = source_fixture(tmp_path)
    prepared = tmp_path / 'prepared'
    prepare_review(*source, prepared, shard=shard)
    docs = [decisions(tmp_path / f'decisions-{p.stem}.json', p) for p in sorted(prepared.glob('review-*.jsonl'))]
    return source, prepared, docs


def test_full_accepted_collection_preserves_original_histories_structures_and_futures(tmp_path):
    source, prepared, docs = prepare_with_decisions(tmp_path)
    resolved = tmp_path / 'resolved'; resolve_review(prepared, docs, [], resolved)
    output = tmp_path / 'labels'; summary = collect_reviewed(*source, [resolved], output)
    assert summary['independently_accepted_original_annotations'] == 3
    assert summary['by_split'] == {'train': 1, 'validation': 1, 'test': 1}
    assert summary['derived_color_mirror_records'] == 0 and not summary['student_training_started']
    for split in ['train', 'validation', 'test']:
        q = load_jsonl(source[0] / f'{split}.queries.jsonl')[0]
        row = load_jsonl(output / f'{split}.jsonl')[0]
        value = parse_explanation(row['answer'])
        for field in ['game_id', 'split', 'initial_fen', 'moves', 'history', 'fen', 'feature_key', 'future_moves']:
            assert row[field] == q['record'][field]
        assert row['future_branches'] == [value['pv'], *[b['pv'] for b in value['branches']]]
        assert row['prose_review']['changed'] is False
    packet = load_jsonl(prepared / 'review-0.jsonl')[0]
    assert 'recorded_source_headers' not in packet and 'future_moves' not in packet
    assert all('fen_before' in f and 'fen_after' in f for b in packet['branches'] for f in b['line_facts'])


def test_incomplete_author_shard_and_changed_isolation_binding_refuse_before_output(tmp_path):
    source = source_fixture(tmp_path)
    progress = source[0] / 'shards/author-0-progress.json'
    doc = json.loads(progress.read_text()); doc['status'] = 'in_progress'; atomic_json(progress, doc)
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='must be complete'):
        prepare_review(*source, output)
    assert not output.exists()
    doc['status'] = 'complete'; atomic_json(progress, doc)
    path = source[0] / 'test.queries.jsonl'; path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='identity differs'):
        prepare_review(*source, output)
    assert not output.exists()


def test_partial_shard_acceptance_cannot_collect_the_full_batch(tmp_path):
    source, prepared, docs = prepare_with_decisions(tmp_path, shard=0)
    resolved = tmp_path / 'resolved'; resolve_review(prepared, docs, [], resolved)
    with pytest.raises(ValueError, match='entire original query batch'):
        collect_reviewed(*source, [resolved], tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('problem', ['self_review', 'foreign_id', 'wrong_hash', 'packet_hash', 'accept_with_issues'])
def test_semantic_decisions_require_independence_and_exact_packet_annotation_bindings(tmp_path, problem):
    _, prepared, docs = prepare_with_decisions(tmp_path)
    path = docs[0]; doc = json.loads(path.read_text())
    if problem == 'self_review':doc['reviewer_agent'] = load_jsonl(prepared / 'review-0.jsonl')[0]['teacher_author_agent']
    if problem == 'foreign_id':doc['results'][0].update(id='foreign', reviewed_annotation_sha256=None)
    if problem == 'wrong_hash':doc['results'][0]['reviewed_annotation_sha256'] = 'different'
    if problem == 'packet_hash':doc['input_packet_sha256'] = 'different'
    if problem == 'accept_with_issues':doc['results'][0]['issues'] = ['unresolved factual issue']
    atomic_json(path, doc)
    with pytest.raises(ValueError):resolve_review(prepared, docs, [], tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


def test_reject_requires_exact_revision_ancestry_and_independent_acceptance(tmp_path):
    source, prepared, docs = prepare_with_decisions(tmp_path)
    packets_path = prepared / 'review-0.jsonl'; packet = load_jsonl(packets_path)[0]
    decisions(docs[0], packets_path, {packet['id']: 'reject'})
    with pytest.raises(ValueError, match='independently accepted'):
        resolve_review(prepared, docs, [], tmp_path / 'not-created')
    repair = copy.deepcopy(packet)
    repair['teacher_annotation'].update(explanation=PROSE + '原始拒收及修订均保留，不覆盖旧文件。',
        corrected_from_annotation_sha256=packet['reviewed_annotation_sha256'],
        correction_review_sha256=digest(docs[0]))
    repair['reviewed_annotation_sha256'] = annotation_hash(repair['teacher_annotation'])
    repairs = tmp_path / 'repair-packets.jsonl'; write_jsonl(repairs, [repair])
    acceptance = decisions(tmp_path / 'repair-acceptance.json', repairs)
    selected, _, _, _ = resolved_packets(prepared, [*docs, acceptance], [repairs])
    assert len(selected) == 3 and repair['teacher_annotation'] in selected
    resolved = tmp_path / 'resolved'; resolve_review(prepared, [*docs, acceptance], [repairs], resolved)
    summary = collect_reviewed(*source, [resolved], tmp_path / 'labels')
    assert summary['independently_accepted_original_annotations'] == 3
    assert json.loads((resolved / 'manifest.json').read_text())['verification']['accepted_unchanged_prose'] == 2
    repair['teacher_annotation']['corrected_from_annotation_sha256'] = 'unknown-parent'
    repair['reviewed_annotation_sha256'] = annotation_hash(repair['teacher_annotation'])
    write_jsonl(repairs, [repair]); decisions(acceptance, repairs)
    with pytest.raises(ValueError, match='exact rejected ancestor'):
        resolved_packets(prepared, [*docs, acceptance], [repairs])


def test_repair_cannot_rewrite_native_facts_or_bypass_a_conflicting_rejection(tmp_path):
    _, prepared, docs = prepare_with_decisions(tmp_path)
    packets_path = prepared / 'review-0.jsonl'; packet = load_jsonl(packets_path)[0]
    reject = decisions(tmp_path / 'contradictory-review.json', packets_path, {packet['id']: 'reject'})
    with pytest.raises(ValueError, match='independently accepted'):
        resolved_packets(prepared, [*docs, reject])
    repair = copy.deepcopy(packet); repair['board']['a0'] = '虚构棋子'
    repairs = tmp_path / 'changed-facts.jsonl'; write_jsonl(repairs, [repair])
    with pytest.raises(ValueError, match='changes its original facts'):
        resolved_packets(prepared, docs, [repairs])


def test_collector_rechecks_resolved_bundle_bytes_and_prepared_original_annotations(tmp_path):
    source, prepared, docs = prepare_with_decisions(tmp_path)
    resolved = tmp_path / 'resolved'; resolve_review(prepared, docs, [], resolved)
    path = resolved / 'annotations.jsonl'; path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='identity differs'):
        collect_reviewed(*source, [resolved], tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


def test_recorded_teacher_requires_substantive_chinese_prose(tmp_path):
    source = source_fixture(tmp_path)
    q = load_jsonl(source[0] / 'train.queries.jsonl')[0]
    with pytest.raises(ValueError, match='80 Chinese'):
        review_packet(q, {'id': q['id'], 'explanation': '中' + 'English text ' * 20, **TEACHER}, '/root/author')


def test_repair_preparation_preserves_rejection_bytes_and_resolves_exact_ancestry(tmp_path):
    source, prepared, docs = prepare_with_decisions(tmp_path)
    packets_path = prepared / 'review-0.jsonl'; packet = load_jsonl(packets_path)[0]
    decisions(docs[0], packets_path, {packet['id']: 'reject'})
    output = tmp_path / 'repair-inputs'; summary = prepare_repairs([prepared], docs, output)
    assert summary['prepared_rejected_original_annotations'] == 1
    assert summary['prepared_original_packet_scope'] == 3
    assert not summary['teacher_repairs_or_acceptances_generated']
    query = load_jsonl(output / 'repair-queries-0.jsonl')[0]
    assert query['original_packet'] == packet
    snapshot = output / 'original-review-0-snapshot.json'
    assert snapshot.read_bytes() == docs[0].read_bytes()
    assert query['correction_review_sha256'] == digest(snapshot) == digest(docs[0])
    assert query['corrected_from_annotation_sha256'] == packet['reviewed_annotation_sha256']
    assert 'future_moves' not in query['original_packet'] and 'recorded_source_headers' not in query['original_packet']
    repair = copy.deepcopy(query['original_packet'])
    repair['teacher_annotation'].update(explanation=PROSE + '此处是受控合同修订，保留准确的拒收祖先。',
        corrected_from_annotation_sha256=query['corrected_from_annotation_sha256'],
        correction_review_sha256=query['correction_review_sha256'])
    repair['reviewed_annotation_sha256'] = annotation_hash(repair['teacher_annotation'])
    repairs = tmp_path / 'repaired.jsonl'; write_jsonl(repairs, [repair])
    acceptance = decisions(tmp_path / 'repair-accepted.json', repairs)
    snapshots = sorted(output.glob('original-review-*-snapshot.json'))
    with pytest.raises(ValueError, match='independently accepted'):
        resolved_packets(prepared, snapshots, [repairs])
    resolved = tmp_path / 'resolved'; resolve_review(prepared, [*snapshots, acceptance], [repairs], resolved)
    assert collect_reviewed(*source, [resolved], tmp_path / 'collected')['original_teacher_annotations'] == 3


@pytest.mark.parametrize('problem', ['self_review', 'wrong_annotation_hash', 'wrong_packet_hash', 'empty_rejection_issues'])
def test_repair_preparation_rejects_unbound_or_unactionable_neural_decisions(tmp_path, problem):
    _, prepared, docs = prepare_with_decisions(tmp_path)
    packets_path = prepared / 'review-0.jsonl'; packet = load_jsonl(packets_path)[0]
    decisions(docs[0], packets_path, {packet['id']: 'reject'})
    doc = json.loads(docs[0].read_text())
    if problem == 'self_review':doc['reviewer_agent'] = packet['teacher_author_agent']
    if problem == 'wrong_annotation_hash':doc['results'][0]['reviewed_annotation_sha256'] = 'wrong'
    if problem == 'wrong_packet_hash':doc['input_packet_sha256'] = 'wrong'
    if problem == 'empty_rejection_issues':doc['results'][0]['issues'] = [' ']
    atomic_json(docs[0], doc)
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError):prepare_repairs([prepared], docs, output)
    assert not output.exists()


def test_repair_preparation_recomputes_facts_even_if_a_packet_is_rebound(tmp_path):
    _, prepared, docs = prepare_with_decisions(tmp_path)
    path = prepared / 'review-0.jsonl'; packet = load_jsonl(path)[0]
    packet['board']['e0'] = '虚构棋子'; write_jsonl(path, [packet])
    proof = json.loads((prepared / 'manifest.json').read_text())
    proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(prepared / 'manifest.json', proof)
    decisions(docs[0], path, {packet['id']: 'reject'})
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='native board or line facts'):
        prepare_repairs([prepared], docs, output)
    assert not output.exists()


def test_repair_preparation_does_not_turn_accepted_prose_into_repair_labels(tmp_path):
    _, prepared, docs = prepare_with_decisions(tmp_path)
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='No independently rejected'):
        prepare_repairs([prepared], docs, output)
    assert not output.exists()
    with pytest.raises(ValueError, match='repeat original'):
        prepare_repairs([prepared, prepared], docs, output)
    assert not output.exists()


def test_serial_and_spawned_repair_preparation_preserve_native_queries_and_review_bytes(tmp_path):
    _, prepared, docs = prepare_with_decisions(tmp_path)
    for doc in docs:
        packet_path = next(p for p in prepared.glob('review-*.jsonl') if digest(p) == json.loads(doc.read_text())['input_packet_sha256'])
        packet = load_jsonl(packet_path)[0]; decisions(doc, packet_path, {packet['id']: 'reject'})
    a, b = tmp_path / 'repair-serial', tmp_path / 'repair-parallel'
    assert prepare_repairs([prepared], docs, a) == prepare_repairs([prepared], docs, b, workers=2)
    for path in a.glob('original-review-*-snapshot.json'):
        assert path.read_bytes() == (b / path.name).read_bytes()
    for path in a.glob('repair-queries-*.jsonl'):
        left = path.read_text().replace(str(a), '<OUTPUT>')
        right = (b / path.name).read_text().replace(str(b), '<OUTPUT>')
        assert left == right


def test_serial_and_spawned_review_preparation_and_collection_preserve_bytes(tmp_path):
    source, prepared, docs = prepare_with_decisions(tmp_path)
    parallel = tmp_path / 'parallel-prepared'; prepare_review(*source, parallel, workers=2)
    for index in range(3):
        assert (prepared / f'review-{index}.jsonl').read_bytes() == (parallel / f'review-{index}.jsonl').read_bytes()
    resolved = tmp_path / 'resolved'; resolve_review(prepared, docs, [], resolved)
    a, b = tmp_path / 'serial-labels', tmp_path / 'parallel-labels'
    assert collect_reviewed(*source, [resolved], a, workers=1) == collect_reviewed(*source, [resolved], b, workers=2)
    for split in ['train', 'validation', 'test']:
        assert (a / f'{split}.jsonl').read_bytes() == (b / f'{split}.jsonl').read_bytes()


@pytest.mark.parametrize('phase', ['prepare', 'collect', 'prepare-repairs'])
def test_inputs_changed_during_native_processing_refuse_before_writing(tmp_path, monkeypatch, phase):
    from xqgeneral import recorded_coach_prose as pipeline
    source, prepared, docs = prepare_with_decisions(tmp_path)
    resolved = tmp_path / 'resolved'; resolve_review(prepared, docs, [], resolved)
    name = {'prepare': 'review_packet_item', 'collect': 'reviewed_label_item',
            'prepare-repairs': 'repair_query_item'}[phase]
    original = getattr(pipeline, name)
    target = source[0] / 'shards/annotations-0.jsonl' if phase == 'prepare' else docs[0]
    def mutate_after_processing(item):
        result = original(item)
        target.write_text(target.read_text() + '\n')
        return result
    monkeypatch.setattr(pipeline, name, mutate_after_processing)
    output = tmp_path / 'not-created'
    if phase == 'prepare-repairs':
        path = prepared / 'review-0.jsonl'; packet = load_jsonl(path)[0]
        decisions(docs[0], path, {packet['id']: 'reject'})
    with pytest.raises(ValueError, match='bound input changed'):
        if phase == 'prepare':prepare_review(*source, output)
        elif phase == 'collect':collect_reviewed(*source, [resolved], output)
        else:prepare_repairs([prepared], docs, output)
    assert not output.exists()
