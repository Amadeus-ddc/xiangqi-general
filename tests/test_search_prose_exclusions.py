"""Review exclusions preserve raw production and are reconstructed before training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from test_merge_search import rebind, setup_merge
from test_search_collection import collect_args, recorded_collection
from xqgeneral import collect_search, merge_search
from xqgeneral.evidence import atomic_json, digest, load_jsonl, write_jsonl
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.search_prose_exclusions import PROFILE, exclude_reviewed_prose


def review_for(queries, responses, rows):
    return dict(profile=PROFILE, queries_sha256=digest(queries), responses_sha256=digest(responses),
        reviewer='Explicit CPU test fixture, no neural or human review', exclusions=[dict(
            id=row['id'], explanation_sha256=hashlib.sha256(json.loads(row['answer'])['explanation'].encode()).hexdigest(),
            reason='CPU fixture exclusion checks exact provenance and never claims a real semantic rating') for row in rows])


def simple_case(tmp_path):
    queries = tmp_path / 'queries.jsonl'; queries.write_text('CPU fixture queries\n')
    responses = tmp_path / 'responses.jsonl'; responses.write_text('CPU fixture raw responses\n')
    rows = [dict(id='case-' + str(i), answer=json.dumps(dict(explanation='中文原始正文' + str(i)))) for i in range(3)]
    return queries, responses, rows, review_for(queries, responses, rows[1:2])


def test_only_remove_reviewed_case_without_mutating_rows_or_earlier_rejections(tmp_path):
    qp, rp, rows, review = simple_case(tmp_path)
    prior = [dict(id='native-rejected', reason='Original native legality error')]
    before = deepcopy((rows, prior, review))
    kept, rejected, note = exclude_reviewed_prose(rows, prior, review,
        queries_sha256=digest(qp), responses_sha256=digest(rp))
    assert kept == [rows[0], rows[2]] and kept[0] is rows[0]
    assert rejected[0] == prior[0] and rejected[1]['id'] == 'case-1'
    assert 'CPU fixture exclusion' in rejected[1]['reason']
    assert (rows, prior, review) == before
    assert note['prose_review_excluded_training_labels'] == 1
    assert note['remaining_prose_semantics_verified_by_exclusions'] is False


@pytest.mark.parametrize('problem', ['different_queries', 'different_responses', 'changed_prose',
    'unknown_id', 'duplicate_id', 'empty', 'boolean_list', 'empty_reason', 'missing_hash',
    'wrong_profile', 'unknown_field', 'no_reviewer'])
def test_stale_or_unbound_reviews_cannot_change_native_dataset(tmp_path, problem):
    qp, rp, rows, review = simple_case(tmp_path)
    if problem == 'different_queries': review['queries_sha256'] = 'a' * 64
    elif problem == 'different_responses': review['responses_sha256'] = 'b' * 64
    elif problem == 'changed_prose': rows[1]['answer'] = json.dumps(dict(explanation='changed'))
    elif problem == 'unknown_id': review['exclusions'][0]['id'] = 'foreign'
    elif problem == 'duplicate_id': review['exclusions'] *= 2
    elif problem == 'empty': review['exclusions'] = []
    elif problem == 'boolean_list': review['exclusions'] = True
    elif problem == 'empty_reason': review['exclusions'][0]['reason'] = ' '
    elif problem == 'missing_hash': review['exclusions'][0].pop('explanation_sha256')
    elif problem == 'wrong_profile': review['profile'] = 'unrecognized'
    elif problem == 'unknown_field': review['automatically_verified_everything'] = True
    elif problem == 'no_reviewer': review['reviewer'] = ''
    with pytest.raises(ValueError):
        exclude_reviewed_prose(rows, [], review, queries_sha256=digest(qp), responses_sha256=digest(rp))


def recollect(tmp_path, monkeypatch, leaf, drop_count=1):
    p = json.loads((leaf / 'manifest.json').read_text())['config']
    rows = load_jsonl(leaf / 'train.jsonl')
    review_path = tmp_path / 'review.json'
    atomic_json(review_path, review_for(p['queries'], p['responses'], rows[:drop_count]))
    output = tmp_path / 'reviewed-labels'
    before = {path: digest(path) for path in [Path(p['queries']), Path(p['responses'])]}
    monkeypatch.setattr(sys, 'argv', ['collect_search', '--queries', p['queries'], '--responses', p['responses'],
        '--config', p['config'], '--heldout-data', p['heldout_data'], '--validation-data', p['validation_data'],
        '--output', str(output), '--prose-review-exclusions', str(review_path)])
    collect_search.main()
    assert all(digest(path) == value for path, value in before.items())
    assert load_jsonl(output / 'train.jsonl') == rows[drop_count:]
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['inputs'][str(review_path)]['sha256'] == digest(review_path)
    assert proof['verification']['remaining_prose_semantics_verified_by_exclusions'] is False
    assert proof['verification']['all_strategic_prose_semantics_verified'] is False
    for split in ('validation', 'test'):
        assert (output / (split + '.jsonl')).read_bytes() == (leaf / (split + '.jsonl')).read_bytes()
    return output, review_path


def test_reviewed_leaf_merges_and_reopens_with_original_raw_teacher_records(tmp_path, monkeypatch):
    source, leaves, _ = setup_merge(tmp_path, monkeypatch)
    reviewed, _ = recollect(tmp_path, monkeypatch, leaves[0])
    dest = tmp_path / 'merged'
    result = merge_search.merge_collections([reviewed, leaves[1]], source, dest)
    assert result['accepted_training_roots'] == 34
    assert result['rejected_consolidations'] == 1
    assert merge_search.collected_dataset(dest, source, BoundInputs())['verification'] == result


def test_direct_reviewed_leaf_reconstructs_actual_raw_exclusions_before_training(tmp_path, monkeypatch):
    source, leaves, _ = setup_merge(tmp_path, monkeypatch, leaf_slices=[slice(None)])
    reviewed, _ = recollect(tmp_path, monkeypatch, leaves[0], drop_count=2)
    proof = merge_search.collected_dataset(reviewed, source, BoundInputs())
    assert proof['verification']['accepted_training_roots'] == 33
    assert proof['verification']['prose_review_excluded_training_labels'] == 2


@pytest.mark.parametrize('problem', ['changed_reason', 'changed_explanation_hash', 'lost_binding',
    'changed_rejected', 'restored_excluded_label', 'wrong_summary', 'removed_option'])
def test_reviewed_leaf_tampering_fails_even_if_local_manifest_is_rebound(tmp_path, monkeypatch, problem):
    source, leaves, _ = setup_merge(tmp_path, monkeypatch, leaf_slices=[slice(None)])
    reviewed, review_path = recollect(tmp_path, monkeypatch, leaves[0], drop_count=2)
    manifest_path = reviewed / 'manifest.json'
    proof = json.loads(manifest_path.read_text())
    if problem in ('changed_reason', 'changed_explanation_hash'):
        review = json.loads(review_path.read_text())
        review['exclusions'][0]['reason' if problem == 'changed_reason' else 'explanation_sha256'] = 'different'
        atomic_json(review_path, review); rebind(manifest_path, 'inputs', review_path)
    elif problem == 'lost_binding':
        proof['inputs'].pop(str(review_path)); atomic_json(manifest_path, proof)
    elif problem == 'changed_rejected':
        atomic_json(reviewed / 'rejected.json', []); rebind(manifest_path, 'outputs', reviewed / 'rejected.json')
    elif problem == 'restored_excluded_label':
        rows = load_jsonl(leaves[0] / 'train.jsonl')
        write_jsonl(reviewed / 'train.jsonl', rows); rebind(manifest_path, 'outputs', reviewed / 'train.jsonl')
    elif problem == 'wrong_summary':
        proof['verification']['prose_review_excluded_training_labels'] = 1; atomic_json(manifest_path, proof)
    elif problem == 'removed_option':
        proof['config'].pop('prose_review_exclusions'); atomic_json(manifest_path, proof)
    with pytest.raises(ValueError):
        merge_search.collected_dataset(reviewed, source, BoundInputs())


def test_excluding_only_native_label_preserves_failure_and_writes_no_completed_dataset(tmp_path, monkeypatch):
    args, _, miner, query = recorded_collection(tmp_path)
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner))
    collect_search.main()
    review_path = tmp_path / 'review.json'
    atomic_json(review_path, review_for(miner / 'queries.jsonl', tmp_path / 'responses.jsonl',
        load_jsonl(tmp_path / 'accepted/train.jsonl')))
    dest = tmp_path / 'empty-reviewed'
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner,
        output=dest, **{'prose-review-exclusions': review_path}))
    with pytest.raises(RuntimeError, match='empty dataset'):
        collect_search.main()
    assert not (dest / 'manifest.json').exists()
    assert json.loads((dest / 'rejected.json').read_text())[0]['id'] == query['id']
