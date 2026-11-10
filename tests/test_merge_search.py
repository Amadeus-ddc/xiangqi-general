"""Real native pool/labels; controlled production receipts, no neural inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from test_search_collection import recorded_collection, response, TEACHER
from xqgeneral import collect_search, distillation_preflight, merge_search
from xqgeneral.evidence import atomic_json, digest, iter_jsonl, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.explanations import move_facts
from xqgeneral.recorded_search_inputs import BoundInputs, prepare, prepared_roots
from xqgeneral.rules import legal_moves, play
from xqgeneral.search_critical import acceptance_evidence


def rebind(proof_path, section, path):
    proof = json.loads(proof_path.read_text())
    proof[section][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(proof_path, proof)


def setup_merge(tmp_path, monkeypatch, unique_count=35, leaf_slices=None):
    args, _, _, _ = recorded_collection(tmp_path)
    pool = tmp_path / 'merge-pool'
    counts = prepare(**args, output=pool, min_ply=1, per_game=16)
    total = counts['candidate_training_histories']
    roots, forbidden = prepared_roots(pool, args['data'], total, 20261013,
        json.loads((args['data'] / 'manifest.json').read_text())['outputs'])
    queries = []
    for root in roots:
        choices = [m for m in sorted(legal_moves(root['fen']))
                   if position_key(play(root['fen'], m)) not in forbidden]
        if len(choices) < 2:
            continue
        score = {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'}
        queries.append({'id': root['id'] + '-search', 'source_root_id': root['id'], 'depth': 0,
            'record': root, 'target_fields': {'move': choices[0], 'pv': [choices[0]],
            'candidates': choices[:2], 'facts': move_facts(root['fen'], choices[0]), 'evaluation': score,
            'branches': [{'move': m, 'pv': [m], 'evaluation': score} for m in choices[:2]]}})
    assert len(queries) >= 35
    queries = queries[:unique_count]
    model = tmp_path / 'teacher-model'; model.mkdir()
    weights = model / 'weights.manifest.json'
    atomic_json(weights, {**TEACHER, 'all_official_lfs_sha256_matched': True})
    teachers = tmp_path / 'complete-teachers.json'
    teacher = dict(TEACHER, model_path=str(model), tensor_parallel_size=2)
    atomic_json(teachers, {'search_consolidator': teacher})
    worker = {'model_quant_config_is_none': True, 'quant_methods': ['UnquantizedLinearMethod'],
              'parameter_elements_by_dtype': {'torch.bfloat16': 27}}
    identity = dict(TEACHER, backend='vllm', parameter_elements_by_dtype={'torch.bfloat16': 54},
        loaded_worker_identities=[worker, deepcopy(worker)], weight_manifest_sha256=digest(weights))
    source = tmp_path / 'selected.pt'; source.write_bytes(b'controlled full student parent')
    outputs = []
    selections = [queries[:20], queries[17:]] if leaf_slices is None else [queries[part] for part in leaf_slices]
    for index, selected in enumerate(selections):
        leaf = tmp_path / f'leaf-{index}'; leaf.mkdir()
        mining, inference = leaf / 'mining', leaf / 'teacher'
        mining.mkdir(); inference.mkdir()
        qp, rp = mining / 'queries.jsonl', inference / 'responses.jsonl'
        write_jsonl(qp, selected)
        answers = []
        for q in selected:
            raw = response();raw.update(id=q['id'], teacher=identity)
            if index:
                value = json.loads(raw['text']);value['explanation'] += '这里保留第二批的原始正文。'
                raw['text'] = json.dumps(value, ensure_ascii=False)
            answers.append(raw)
        write_jsonl(rp, answers)
        atomic_json(inference / 'deployment.json', identity)
        atomic_json(inference / 'manifest.json', manifest('local_teacher_inference',
            {'input': str(qp), 'config': str(teachers), 'backend': 'vllm'}, [qp, teachers, weights],
            [rp, inference / 'deployment.json'], {'examples': len(answers),
            'neural_teacher_inference_executed': True, 'teacher': identity,
            'controlled_CPU_fixture_not_real_teacher_execution': True}))
        by_id = {q['source_root_id']: q for q in selected}
        attempts = [{'source_root_id': row['id'], 'trace': [],
                     'query': by_id.get(row['id']),
                     'reason': 'accepted_for_consolidation' if row['id'] in by_id else 'recursion_limit'}
                    for row in roots]
        write_jsonl(mining / 'results.partial.jsonl', attempts)
        source_proof = json.loads((tmp_path / 'miner/manifest.json').read_text())
        config = dict(source_proof['config'], checkpoint=str(source), limit=total, recorded_inputs=str(pool))
        verification = dict(source_proof['verification'], accepted_for_consolidation=len(selected),
            processed=len(attempts), additional_reserved_canonical_and_explanation_positions=len(forbidden))
        inputs = [source, *[args['data'] / f'{s}.jsonl' for s in ['train', 'validation', 'test']],
                  *[pool / name for name in ['manifest.json', 'roots.jsonl', 'counts.json', 'heldout-positions.json']]]
        atomic_json(mining / 'manifest.json', manifest('search_distillation_mining', config,
            inputs, [qp, mining / 'results.partial.jsonl'], verification))
        output = leaf / 'labels'
        monkeypatch.setattr(sys, 'argv', ['collect_search', '--queries', str(qp), '--responses', str(rp),
            '--config', str(teachers), '--heldout-data', str(args['data']),
            '--validation-data', str(args['heldout_data']), '--output', str(output)])
        collect_search.main()
        # Original spacing, unicode escapes and line terminators must survive merging.
        train = output / 'train.jsonl'
        train.write_bytes(b''.join((json.dumps(row, ensure_ascii=True, separators=(',', ':')) + '  \r\n').encode()
                                  for row in iter_jsonl(train)))
        rebind(output / 'manifest.json', 'outputs', train)
        outputs.append(output)
    return source, outputs, queries


def set_paper_recursive(leaf):
    path = leaf.parent / 'mining/manifest.json';value = json.loads(path.read_text())
    value['config'].update(acceptance_policy='paper_recursive', min_clean_ply=1,
        child_contract='move_eval', max_root_retries=3, root_candidate_policy='paper_fill')
    value['verification'].update(acceptance_policy='paper_recursive', **acceptance_evidence('paper_recursive'))
    atomic_json(path, value);rebind(leaf / 'manifest.json', 'inputs', path)


@pytest.mark.parametrize('policy', ['strict', 'paper_recursive'])
def test_two_small_real_native_leaves_merge_by_full_history_keep_bytes_and_reopen(tmp_path, monkeypatch, policy):
    source, leaves, queries = setup_merge(tmp_path, monkeypatch)
    if policy == 'paper_recursive':
        for leaf in leaves: set_paper_recursive(leaf)
    with pytest.raises(ValueError):
        distillation_preflight.collected_dataset(leaves[0], source, BoundInputs())
    before = {str(p): digest(p) for leaf in leaves for p in leaf.glob('*') if p.is_file()}
    output = tmp_path / 'merged'
    result = merge_search.merge_collections(leaves, source, output)
    assert result['raw_accepted_training_records'] == 38
    assert result['accepted_training_roots'] == result['unique_full_history_contexts'] == 35
    assert result['duplicate_history_records'] == result['duplicate_records_with_different_answers'] == 3
    assert result['unique_attempted_starting_roots'] == len(load_jsonl(tmp_path / 'merge-pool/roots.jsonl'))
    assert result['unique_accepted_starting_roots'] == 35
    assert result['search_acceptance_profile'] == {'policy': policy, 'min_clean_ply': 1}
    assert result['game_overlap'] == result['root_and_future_overlap'] == 0
    assert not result['student_SGD_executed'] and result['new_model_teacher_or_engine_queries'] == 0
    assert (output / 'train.jsonl').read_bytes() == (leaves[0] / 'train.jsonl').read_bytes() + b''.join(
        (leaves[1] / 'train.jsonl').read_bytes().splitlines(keepends=True)[3:])
    assert all(digest(Path(p)) == value for p, value in before.items())
    for split in ['validation', 'test']:
        assert (output / f'{split}.jsonl').read_bytes() == (leaves[0] / f'{split}.jsonl').read_bytes()
    proof = distillation_preflight.collected_dataset(output, source, BoundInputs())
    assert proof['kind'] == merge_search.MERGED_KIND and proof['verification'] == result
    provenance = load_jsonl(output / 'provenance.jsonl')
    assert len(provenance) == 38 and sum(p['decision'] == 'duplicate' for p in provenance) == 3
    assert [r['search_source_root'] for r in iter_jsonl(output / 'train.jsonl')] == [q['source_root_id'] for q in queries]
    with pytest.raises(FileExistsError):
        merge_search.merge_collections(leaves, source, output)


def test_global_floor_counts_unique_histories_after_cross_collection_deduplication(tmp_path, monkeypatch):
    source, leaves, _ = setup_merge(tmp_path, monkeypatch, unique_count=31)
    assert sum(len(load_jsonl(p / 'train.jsonl')) for p in leaves) == 34
    with pytest.raises(ValueError, match='32 unique'):
        merge_search.merge_collections(leaves, source, tmp_path / 'too-small-after-deduplication')


@pytest.mark.parametrize('problem', ['alias', 'too_small', 'wrong_parent', 'different_teacher', 'pending_teacher',
    'quantized_worker', 'wrong_teacher_query', 'wrong_weight_inventory', 'changed_rejections', 'changed_answer',
    'changed_query_ledger', 'missing_raw_input', 'different_heldout', 'nested', 'changed_child', 'changed_provenance',
    'mismatched_acceptance', 'zero_clean_ply', 'boolean_clean_ply'])
def test_source_or_merged_tampering_cannot_become_qualified_training_data(tmp_path, monkeypatch, problem):
    source, leaves, _ = setup_merge(tmp_path, monkeypatch)
    output = tmp_path / 'must-not-qualify'
    data = leaves[1];proof_path = data / 'manifest.json';proof = json.loads(proof_path.read_text())
    teacher_path = data.parent / 'teacher/manifest.json'
    if problem in ['nested', 'changed_child', 'changed_provenance']:
        merge_search.merge_collections(leaves, source, output)
        if problem == 'nested':
            with pytest.raises(ValueError):
                merge_search.merge_collections([output, leaves[0]], source, tmp_path / 'nested-output')
            return
        if problem == 'changed_child':
            path = data / 'train.jsonl';path.write_bytes(path.read_bytes() + b'changed')
        else:
            path = output / 'provenance.jsonl';rows = load_jsonl(path);rows[20]['decision'] = 'kept'
            write_jsonl(path, rows);rebind(output / 'manifest.json', 'outputs', path)
        with pytest.raises(ValueError):
            distillation_preflight.collected_dataset(output, source, BoundInputs())
        return
    if problem == 'alias': leaves.append(data / '..' / 'labels')
    elif problem == 'too_small': leaves = leaves[:1]
    elif problem == 'mismatched_acceptance': set_paper_recursive(data)
    elif problem in ['zero_clean_ply', 'boolean_clean_ply']:
        path = data.parent / 'mining/manifest.json';value = json.loads(path.read_text())
        value['config']['min_clean_ply'] = 0 if problem == 'zero_clean_ply' else False
        atomic_json(path, value);rebind(proof_path, 'inputs', path)
    elif problem == 'wrong_parent':
        source = tmp_path / 'other.pt';source.write_bytes(b'changed parent')
    elif problem in ['different_teacher', 'pending_teacher', 'wrong_teacher_query']:
        value = json.loads(teacher_path.read_text())
        if problem == 'different_teacher': value['verification']['teacher']['revision'] = 'other'
        elif problem == 'pending_teacher': value['status'] = 'running'
        else: value['config']['input'] = 'foreign/queries.jsonl'
        atomic_json(teacher_path, value)
    elif problem == 'quantized_worker':
        path = data.parent / 'teacher/deployment.json';value = json.loads(path.read_text())
        value['loaded_worker_identities'][1]['model_quant_config_is_none'] = False
        atomic_json(path, value);rebind(teacher_path, 'outputs', path)
        teacher_proof = json.loads(teacher_path.read_text());teacher_proof['verification']['teacher'] = value
        atomic_json(teacher_path, teacher_proof)
        path = data.parent / 'teacher/responses.jsonl';rows = load_jsonl(path)
        for row in rows: row['teacher'] = value
        write_jsonl(path, rows);rebind(teacher_path, 'outputs', path);rebind(proof_path, 'inputs', path)
    elif problem == 'wrong_weight_inventory':
        path = tmp_path / 'teacher-model/weights.manifest.json';value = json.loads(path.read_text())
        value['revision'] = 'foreign';atomic_json(path, value)
    elif problem in ['changed_rejections', 'changed_answer']:
        path = data / ('rejected.json' if problem == 'changed_rejections' else 'train.jsonl')
        if problem == 'changed_rejections': atomic_json(path, [{'id': 'invented', 'reason': 'invented'}])
        else:
            rows = load_jsonl(path);value = json.loads(rows[0]['answer'])
            value['explanation'] += '这是未经原始教师回答支持的额外改写。'
            rows[0]['answer'] = json.dumps(value, ensure_ascii=False);write_jsonl(path, rows)
        rebind(proof_path, 'outputs', path)
    elif problem == 'changed_query_ledger':
        path = data.parent / 'mining/results.partial.jsonl';rows = load_jsonl(path)
        rows[0]['source_root_id'] = rows[1]['source_root_id'];write_jsonl(path, rows)
        mp = data.parent / 'mining/manifest.json';rebind(mp, 'outputs', path);rebind(proof_path, 'inputs', mp)
    elif problem == 'missing_raw_input':
        proof['inputs'].pop(proof['config']['responses']);atomic_json(proof_path, proof)
    else:
        path = data / 'validation.jsonl';path.write_bytes(path.read_bytes() + b'\n')
        rebind(proof_path, 'outputs', path)
    with pytest.raises(ValueError):
        merge_search.merge_collections(leaves, source, output)
    assert not output.exists()


def test_cli_opens_the_configured_verified_input_session(monkeypatch):
    from contextlib import contextmanager
    from xqgeneral import verified_inputs
    entered, called = [], []
    @contextmanager
    def session():
        entered.append(True)
        yield
    monkeypatch.setattr(verified_inputs, 'input_session_from_environment', session)
    def merge(*args):
        assert entered
        called.append(args)
        return {'controlled_CPU_only': True}
    monkeypatch.setattr(merge_search, 'merge_collections', merge)
    monkeypatch.setattr(sys, 'argv', ['merge_search', '--collections', 'a', 'b', '--checkpoint', 'selected.pt', '--output', 'out'])
    merge_search.main()
    assert called == [(['a', 'b'], 'selected.pt', 'out')]
