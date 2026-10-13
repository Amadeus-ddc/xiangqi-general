"""Reuse the actual complete native check, and reject changed source/label bytes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from xqgeneral import paper_curriculum
from xqgeneral.evidence import atomic_json, digest, load_jsonl
from xqgeneral.foundation_readback import checked_course_sources
from test_paper_curriculum import build, source_fixture


@pytest.fixture
def completed(tmp_path):
    roots, proof, _ = source_fixture(tmp_path)
    data = tmp_path / 'data'
    build([roots], [proof], data, seed=7)
    readback = tmp_path / 'complete-gold'
    paper_curriculum.readback(data, readback)
    corpus = json.loads((data / 'manifest.json').read_text())
    contexts = {}
    for split in ('train', 'validation', 'test'):
        for row in load_jsonl(data / (split + '.jsonl')):
            contexts.setdefault(row['feature_key'], row)
    return data, roots, readback / 'manifest.json', corpus, contexts


def test_explicit_reuse_avoids_the_second_native_regeneration(completed, tmp_path, monkeypatch):
    data, _, proof, corpus, contexts = completed
    calls = []
    original = paper_curriculum.readback

    def observed(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(paper_curriculum, 'readback', observed)
    checked_course_sources(data, corpus, contexts.values(), set(contexts), len(contexts),
                           tmp_path / 'default-still-regenerates')
    assert len(calls) == 1
    calls.clear()
    keys, result, inputs = checked_course_sources(data, corpus, contexts.values(), set(contexts), len(contexts),
        tmp_path / 'unused-source-output', completed_question_readback=proof)
    assert keys == set(contexts) and calls == []
    assert result['paper_questions_regenerated'] == 300
    assert result['paper_question_readback_reused'] is True
    assert result['paper_question_regeneration_executed_this_readback'] is False
    assert result['paper_question_readback_manifest_sha256'] == digest(proof)
    assert proof.resolve() in set(map(Path, inputs))
    assert not (tmp_path / 'unused-source-output').exists()


@pytest.mark.parametrize('target', ['source', 'question', 'verification', 'data_manifest'])
def test_same_size_changes_are_rejected_by_fresh_hashes(completed, target):
    from xqgeneral.paper_readback_reuse import verify_completed_readback
    data, roots, manifest, _, _ = completed
    path = {'source': roots, 'question': data / 'train.jsonl',
            'verification': manifest.parent / 'verification.json', 'data_manifest': data / 'manifest.json'}[target]
    raw = bytearray(path.read_bytes())
    at = next(i for i, b in enumerate(raw) if b in b'0123456789')
    raw[at] = ord('8') if raw[at] != ord('8') else ord('7')
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='changed|binding|verification|identity'):
        verify_completed_readback(data, manifest)


def test_rehashed_wrong_label_corpus_cannot_borrow_a_prior_gold_proof(completed):
    from xqgeneral.paper_readback_reuse import verify_completed_readback
    data, _, proof, corpus, _ = completed
    path = data / 'train.jsonl'
    values = load_jsonl(path)
    values[0]['answer'] = 'wrong-answer-that-was-never-natively-verified'
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))
    corpus['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(data / 'manifest.json', corpus)
    with pytest.raises(ValueError, match='changed|binding'):
        verify_completed_readback(data, proof)


@pytest.mark.parametrize('change', ['status', 'kind', 'counts', 'full_check', 'code', 'missing_input', 'wrong_data'])
def test_incomplete_or_other_corpus_proofs_are_rejected(completed, change):
    from xqgeneral.paper_readback_reuse import verify_completed_readback
    data, _, manifest, _, _ = completed
    proof = json.loads(manifest.read_text())
    if change == 'status':
        proof['status'] = 'building'
    elif change == 'kind':
        proof['kind'] = 'partial_sample_readback'
    elif change == 'counts':
        proof['verification']['questions_regenerated_and_compared'] -= 1
    elif change == 'full_check':
        proof['verification']['all_source_and_question_bytes_checked'] = False
    elif change == 'code':
        proof['code']['revision'] = 'different-producer-revision'
    elif change == 'missing_input':
        proof['inputs'].pop(str(data / 'test.jsonl'))
    else:
        proof['config']['data'] = str(data.parent / 'other-corpus')
    atomic_json(manifest, proof)
    with pytest.raises(ValueError):
        verify_completed_readback(data, manifest)


def test_verified_borrowing_carries_all_transitive_artifacts_and_detects_later_drift(completed):
    from xqgeneral.paper_readback_reuse import verify_completed_readback
    data, roots, manifest, corpus, _ = completed
    verified = verify_completed_readback(data, manifest)
    expected = {manifest.resolve(), (manifest.parent / 'verification.json').resolve(),
                (data / 'manifest.json').resolve(), *(Path(p).resolve() for p in corpus['inputs']),
                *(Path(p).resolve() for p in corpus['outputs'])}
    assert set(verified.inputs) == expected
    verified.check_unchanged()
    roots.write_bytes(roots.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed'):
        verified.check_unchanged()


def test_reuse_still_checks_full_context_and_cache_identity(completed, tmp_path):
    data, _, proof, corpus, contexts = completed
    bad = deepcopy(list(contexts.values()))
    bad[0]['feature_key'] = 'incorrect-feature-key'
    with pytest.raises(ValueError, match='identity'):
        checked_course_sources(data, corpus, bad, set(contexts), len(contexts), tmp_path / 'never',
                               completed_question_readback=proof)


def test_catalog_budget_frequency_and_shortfall_proof_are_reused_together(tmp_path, monkeypatch):
    from test_native_catalog_courses import prepared
    from test_course_rendering import options
    from xqgeneral.paper_readback_reuse import verify_completed_readback
    _, roots, source, catalog_proof, reference = prepared(tmp_path)
    data = tmp_path / 'catalog-data'
    produced = paper_curriculum.build([roots], [source], data, reference_data=[reference],
                                      catalog_proof=catalog_proof, **options())
    output = tmp_path / 'catalog-gold'
    complete = paper_curriculum.readback(data, output)
    proof_path = output / 'manifest.json'
    receipt = verify_completed_readback(data, proof_path)
    assert receipt.verification == complete
    assert receipt.verification['task_position_selection'] == produced['task_position_selection']
    assert receipt.verification['answer_class_frequency_by_split_stage_task'] == produced['answer_class_frequency_by_split_stage_task']
    assert catalog_proof.resolve() in set(receipt.inputs)
    changed = json.loads(proof_path.read_text())
    changed['verification'].pop('task_position_selection')
    atomic_json(proof_path, changed)
    with pytest.raises(ValueError, match='sampling|budgets|shortfall'):
        verify_completed_readback(data, proof_path)


def test_removed_artifact_after_verification_is_refused(completed):
    from xqgeneral.paper_readback_reuse import verify_completed_readback
    data, roots, proof, _, _ = completed
    checked = verify_completed_readback(data, proof)
    roots.unlink()
    with pytest.raises(ValueError, match='changed|disappeared'):
        checked.check_unchanged()
