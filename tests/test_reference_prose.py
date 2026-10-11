import hashlib
import json
from pathlib import Path

import pytest

from xqgeneral.evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from xqgeneral.reference_prose import AUDIT_KIND, annotation_hash, correct_references, teacher_annotation
from xqgeneral.reviewed_explanations import REFERENCE_CORRECTION_KIND
from xqgeneral.revise_prose import TEACHER
from test_reviewed_explanations import label, preflight_fixture, source


def decision(identity, sha, line_sha, *, accept=True, reviewer='independent-reviewer', **extra):
    return {'id': identity, 'reviewed_annotation_sha256': sha, 'source_label_line_sha256': line_sha,
            'reviewer': reviewer, **TEACHER, 'factual_correctness': 4 if accept else 3,
            'strategic_reasoning': 4, 'clarity': 4, 'instruction_adherence': 4,
            'unsupported_claims': [] if accept else ['受控夹具的原正文需要修订'],
            'reason': '逐条检查了此夹具中的推荐走法、候选差别和交换边界。',
            'verdict': 'accept' if accept else 'reject', **extra}


def audit_fixture(tmp_path):
    rows = [label('a', 'train', ['b0c2']), label('b', 'validation', ['h0g2']),
            label('c', 'test', ['b2e2'])]
    for r in rows[1:]:r.pop('prose_review')
    data = source(tmp_path / 'data', rows, 'combined_reviewed_explanation_dataset')
    path = data / 'manifest.json';doc = json.loads(path.read_text())
    doc['verification'].update(source_label_bytes_and_original_splits_preserved=True,
        completed_semantic_acceptance_proofs_reused=True, all_training_rows_have_inline_or_supplemental_acceptance=True,
        complete_native_histories_facts_and_all_continuations_revalidated=True,
        game_overlap=0, root_and_future_and_color_overlap=0)
    atomic_json(path, doc)
    audit = tmp_path / 'audit';audit.mkdir()
    packets, original_decisions, references = [], [], []
    for row in rows[1:]:
        raw = (data / f'{row["split"]}.jsonl').read_bytes();line_sha = hashlib.sha256(raw).hexdigest()
        annotation = teacher_annotation(row);sha = annotation_hash(annotation)
        packets.append({'id': row['id'], 'teacher_annotation': annotation,
                        'reviewed_annotation_sha256': sha, 'source_label_line_sha256': line_sha})
        original_decisions.append(decision(row['id'], sha, line_sha, accept=row['id'] == 'b'))
        references.append({'id': row['id'], 'source_annotation_sha256': sha,
                           'reviewed_annotation_sha256': sha, 'source_label_line_sha256': line_sha})
    write_jsonl(audit / 'packets.jsonl', packets)
    write_jsonl(audit / 'original-decisions.jsonl', original_decisions)
    correction = {**packets[1]['teacher_annotation'],
        'explanation': '这是一段受控数据合同夹具正文，不能视为真实神经教师的质量证据。推荐着法和候选变化保持原有结构事实，评估从根局面的行棋方角度解释，不承诺唯一应着或强制获胜。教学时应同时说明子力交换的代价和继续改善棋子位置的方向。',
        'author_actor': 'different-author', 'corrected_from_annotation_sha256': references[1]['source_annotation_sha256'],
        'correction_review_sha256': digest(audit / 'original-decisions.jsonl'),
        'source_label_line_sha256': references[1]['source_label_line_sha256']}
    write_jsonl(audit / 'corrections.jsonl', [correction]);sha = annotation_hash({k: correction[k] for k in ('id', 'explanation', *TEACHER)})
    final = decision('c', sha, correction['source_label_line_sha256'],
        corrected_from_annotation_sha256=correction['corrected_from_annotation_sha256'],
        correction_review_sha256=correction['correction_review_sha256'],
        author_annotation_file_sha256=digest(audit / 'corrections.jsonl'))
    write_jsonl(audit / 'final-decisions.jsonl', [final]);references[1]['reviewed_annotation_sha256'] = sha
    write_jsonl(audit / 'references.jsonl', references)
    def seal(**flags):
        atomic_json(audit / 'manifest.json', manifest(AUDIT_KIND, {},
            [audit / n for n in ['packets.jsonl', 'original-decisions.jsonl', 'corrections.jsonl', 'final-decisions.jsonl']],
            [audit / 'references.jsonl'], {'human_rating': False, 'student_predictions_seen': False,
                'test_used_for_training_or_selection': False, **flags}))
    seal()
    return data, audit, rows, seal


def test_reference_correction_preserves_training_bytes_unrevised_rows_and_all_engine_facts(tmp_path):
    data, audit, originals, _ = audit_fixture(tmp_path);output = tmp_path / 'corrected'
    before = {s: (data / f'{s}.jsonl').read_bytes() for s in ['train', 'validation', 'test']}
    proof = correct_references(data, audit, output, 1)
    assert (output / 'train.jsonl').read_bytes() == before['train']
    assert (output / 'validation.jsonl').read_bytes() == before['validation']
    assert {s: (data / f'{s}.jsonl').read_bytes() for s in before} == before
    original = originals[2];revised = load_jsonl(output / 'test.jsonl')[0]
    old_answer, new_answer = json.loads(original['answer']), json.loads(revised['answer'])
    assert old_answer['explanation'] != new_answer['explanation']
    assert {k:v for k,v in old_answer.items() if k != 'explanation'} == {k:v for k,v in new_answer.items() if k != 'explanation'}
    assert {k:v for k,v in revised.items() if k not in ['answer', 'prose_review']} == {k:v for k,v in original.items() if k != 'answer'}
    assert proof['audited_heldout_references'] == 2 and proof['revised_heldout_prose_by_split'] == {'test': 1}
    assert proof['training_file_byte_identical'] and not proof['source_label_bytes_and_original_splits_preserved']
    assert not proof['independent_test_used_for_training_or_selection'] and not proof['active_training_inputs_changed']
    assert json.loads((output / 'manifest.json').read_text())['kind'] == REFERENCE_CORRECTION_KIND


def test_serial_and_spawned_native_reference_checks_have_identical_outputs(tmp_path):
    data, audit, _, _ = audit_fixture(tmp_path)
    for workers in [1, 2]:correct_references(data, audit, tmp_path / f'workers-{workers}', workers)
    for filename in ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'verification.json']:
        assert (tmp_path / 'workers-1' / filename).read_bytes() == (tmp_path / 'workers-2' / filename).read_bytes()


def test_relative_manifest_bindings_accept_the_same_absolute_source_files(tmp_path, monkeypatch):
    data, audit, _, _ = audit_fixture(tmp_path)
    for root in [data, audit]:
        path = root / 'manifest.json';doc = json.loads(path.read_text())
        for section in ['inputs', 'outputs']:
            doc[section] = {str(Path(p).relative_to(tmp_path)): value
                            for p, value in doc[section].items()}
        atomic_json(path, doc)
    monkeypatch.chdir(tmp_path)
    result = correct_references(data.resolve(), audit.resolve(), tmp_path / 'corrected', 1)
    assert result['training_file_byte_identical'] and result['revised_heldout_prose'] == 1


@pytest.mark.parametrize('flag', ['human_rating', 'student_predictions_seen', 'test_used_for_training_or_selection'])
def test_human_claims_or_student_informed_heldout_edits_are_refused(tmp_path, flag):
    data, audit, _, seal = audit_fixture(tmp_path);seal(**{flag: True})
    with pytest.raises(ValueError, match='student-blind'):correct_references(data, audit, tmp_path / 'refused', 1)
    assert not (tmp_path / 'refused').exists()


@pytest.mark.parametrize('problem', ['self_review', 'wrong_author_file', 'unsupported_claim', 'inflated_verdict', 'unknown_parent', 'wrong_rejection', 'short_prose'])
def test_accepted_claim_cannot_bypass_independent_exact_revision_ancestry(tmp_path, problem):
    data, audit, _, seal = audit_fixture(tmp_path)
    correction = load_jsonl(audit / 'corrections.jsonl')[0];final = load_jsonl(audit / 'final-decisions.jsonl')[0]
    if problem == 'self_review':final['reviewer'] = correction['author_actor']
    if problem == 'wrong_author_file':final['author_annotation_file_sha256'] = 'f' * 64
    if problem == 'unsupported_claim':final['unsupported_claims'] = ['没有事实支持的保证获胜']
    if problem == 'inflated_verdict':final['factual_correctness'] = 3
    if problem == 'unknown_parent':correction['corrected_from_annotation_sha256'] = 'a' * 64
    if problem == 'wrong_rejection':correction['correction_review_sha256'] = 'a' * 64
    if problem == 'short_prose':correction['explanation'] = '短文不能代替实质教师修订。'
    if problem in ['unknown_parent', 'wrong_rejection', 'short_prose']:
        write_jsonl(audit / 'corrections.jsonl', [correction])
        sha = annotation_hash({k: correction[k] for k in ('id', 'explanation', *TEACHER)})
        final.update(reviewed_annotation_sha256=sha,
            corrected_from_annotation_sha256=correction['corrected_from_annotation_sha256'],
            correction_review_sha256=correction['correction_review_sha256'],
            author_annotation_file_sha256=digest(audit / 'corrections.jsonl'))
        refs = load_jsonl(audit / 'references.jsonl');refs[1]['reviewed_annotation_sha256'] = sha
        write_jsonl(audit / 'references.jsonl', refs)
    write_jsonl(audit / 'final-decisions.jsonl', [final]);seal()
    with pytest.raises(ValueError):correct_references(data, audit, tmp_path / 'refused', 1)
    assert not (tmp_path / 'refused').exists()


@pytest.mark.parametrize('problem', ['partial_scope', 'source_line', 'training_reference', 'changed_artifact'])
def test_foreign_partial_or_tampered_reference_bundles_cannot_modify_labels(tmp_path, problem):
    data, audit, originals, seal = audit_fixture(tmp_path)
    if problem == 'partial_scope':
        refs = load_jsonl(audit / 'references.jsonl');write_jsonl(audit / 'references.jsonl', refs[:1]);seal()
    if problem == 'source_line':
        p = data / 'validation.jsonl';p.write_bytes(b' ' + p.read_bytes())
        doc = json.loads((data / 'manifest.json').read_text());doc['outputs'][str(p)] = {'sha256': digest(p), 'bytes': p.stat().st_size}
        atomic_json(data / 'manifest.json', doc)
    if problem == 'training_reference':
        raw = (data / 'train.jsonl').read_bytes();annotation = teacher_annotation(originals[0]);sha = annotation_hash(annotation);line_sha = hashlib.sha256(raw).hexdigest()
        packets = load_jsonl(audit / 'packets.jsonl');packets.append({'id':'a', 'teacher_annotation':annotation, 'reviewed_annotation_sha256':sha,'source_label_line_sha256':line_sha})
        reviews = load_jsonl(audit / 'original-decisions.jsonl');reviews.append(decision('a',sha,line_sha))
        refs = load_jsonl(audit / 'references.jsonl');refs.append({'id':'a','source_annotation_sha256':sha,'reviewed_annotation_sha256':sha,'source_label_line_sha256':line_sha})
        write_jsonl(audit / 'packets.jsonl',packets);write_jsonl(audit / 'original-decisions.jsonl',reviews);write_jsonl(audit / 'references.jsonl',refs)
        correction = load_jsonl(audit / 'corrections.jsonl')[0]
        correction['correction_review_sha256'] = digest(audit / 'original-decisions.jsonl')
        write_jsonl(audit / 'corrections.jsonl', [correction])
        final = load_jsonl(audit / 'final-decisions.jsonl')[0]
        final.update(correction_review_sha256=correction['correction_review_sha256'],
                     author_annotation_file_sha256=digest(audit / 'corrections.jsonl'))
        write_jsonl(audit / 'final-decisions.jsonl', [final])
        seal()
    if problem == 'changed_artifact':(audit / 'final-decisions.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='no training rows' if problem == 'training_reference' else None):
        correct_references(data, audit, tmp_path / 'refused', 1)
    assert not (tmp_path / 'refused').exists()


def test_reference_correction_preserves_existing_outputs_and_refuses_incomplete_parent(tmp_path):
    data, audit, _, _ = audit_fixture(tmp_path);output = tmp_path / 'existing';output.mkdir();(output / 'keep').write_text('keep')
    with pytest.raises(FileExistsError):correct_references(data,audit,output,1)
    assert (output / 'keep').read_text() == 'keep'
    doc=json.loads((data / 'manifest.json').read_text());doc['verification']['game_overlap']=1;atomic_json(data / 'manifest.json',doc)
    with pytest.raises(ValueError,match='isolation'):correct_references(data,audit,tmp_path / 'refused',1)


def test_token_preflight_accepts_truthful_reference_correction_but_never_tokenizes_test(tmp_path, monkeypatch):
    from xqgeneral.explanation_preflight import preflight
    data, config, completed, _ = preflight_fixture(tmp_path, monkeypatch, long_test=True)
    path=data / 'manifest.json';doc=json.loads(path.read_text());doc['kind']=REFERENCE_CORRECTION_KIND
    doc['verification'].update(training_file_byte_identical=True, heldout_only_prose_and_review_metadata_changed=True,
        identities_structured_labels_histories_and_original_splits_preserved=True,
        completed_parent_native_and_isolation_proofs_reused=True,
        revised_bodies_complete_native_histories_and_continuations_revalidated=True,
        all_original_unrated_heldout_annotations_independently_accepted=True,
        student_predictions_seen=False,independent_test_used_for_training_or_selection=False)
    atomic_json(path,doc)
    result=preflight(data,config,completed,tmp_path / 'preflight',max_tokens=4096,workers=1)
    assert result['independent_test_rows_tokenized']==0 and result['train_validation_rows_tokenized']==2
    assert not result['model_weights_loaded_or_training_started']
    doc['verification']['training_file_byte_identical']=False;atomic_json(path,doc)
    with pytest.raises(ValueError):preflight(data,config,completed,tmp_path / 'refused',max_tokens=4096,workers=1)
    assert not (tmp_path / 'refused').exists()
