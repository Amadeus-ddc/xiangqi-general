import copy
import json
from pathlib import Path

import pytest

from xqgeneral.engine_qualified_explanations import qualify
from xqgeneral.evidence import atomic_json, digest, history_key, manifest
from xqgeneral.explanations import EXPLANATION_QUESTION, move_facts
from xqgeneral.recorded_coach import checked_manifest
from xqgeneral.reviewed_explanations import ENGINE_QUALIFIED_KIND, source_contract
from xqgeneral.revise_prose import TEACHER
from xqgeneral.rules import START_FEN, legal_moves, play, replay


def label(identity, split, moves):
    history = replay(START_FEN, moves)
    fen = history[-1];move = legal_moves(fen)[0]
    pv = [move, legal_moves(play(fen, move))[0]]
    score = {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'}
    value = {'move': move, 'pv': pv, 'candidates': [move], 'evaluation': score,
             'facts': move_facts(fen, move), 'explanation': '受控谱例验证数据合同，不代表真实教师质量。',
             'branches': [{'move': move, 'pv': pv, 'evaluation': score}]}
    return {'id': identity, 'game_id': 'game-' + identity, 'split': split, 'stage': 'explanation',
            'task_type': 'explanation', 'query': {}, 'question': EXPLANATION_QUESTION,
            'initial_fen': START_FEN, 'moves': moves, 'history': history, 'fen': fen,
            'feature_key': history_key(history), 'future_moves': [], 'provenance': 'controlled_contract_fixture',
            'answer': json.dumps(value, ensure_ascii=False), 'teacher': dict(TEACHER),
            'prose_review': {'source_annotation_sha256': 'a' * 64,
                             'reviewed_annotation_sha256': 'b' * 64,
                             'review_manifest_sha256': 'c' * 64, 'changed': False}}


def bound(path):
    return {'sha256': digest(path), 'bytes': path.stat().st_size}


def fixture(root, specs=None, mutate=None, opaque_heldout=False):
    data = root / 'data';reference = root / 'reference'
    data.mkdir();(reference / 'full/rows').mkdir(parents=True)
    rows = [label('a', 'train', ['b0c2']), label('b', 'train', ['h0g2']),
            label('c', 'train', ['b2e2'])]
    heldout = {'validation': label('v', 'validation', ['h2e2']),
               'test': label('t', 'test', ['a0a1'])}
    source_paths = []
    for split in ['train', 'validation', 'test']:
        path = data / f'{split}.jsonl'
        selected = rows if split == 'train' else [heldout[split]]
        raw = b''.join(('  ' + json.dumps(row, ensure_ascii=False) + '  \r\n').encode() for row in selected)
        if opaque_heldout and split != 'train':
            raw = ('preaccepted opaque ' + split + ' bytes\n').encode()
        path.write_bytes(raw);source_paths.append(path)
    verification = {'by_split': {'train': 3, 'validation': 1, 'test': 1},
                    'original_teacher_annotations': 5, 'independently_accepted_original_annotations': 5}
    atomic_json(data / 'manifest.json', manifest('reviewed_recorded_coach_initial_teacher_labels',
        {'purpose': 'controlled acceptance fixture'}, [], source_paths, verification))
    specs = specs or [(0.0, [0.0, 0.0]), (0.0, [0.0, 0.2]), (0.2, [0.2, 0.0])]
    full_counts = {'parse_valid': 3, 'structured_contract_valid': 3, 'first_move_legal': 3,
                   'pv_legal': 3, 'first_move_no_mistake': sum(first < .10 for first, _ in specs),
                   'full_pv_no_mistake': sum(all(x < .10 for x in pv) for _, pv in specs)}
    metrics = {'examples': 3, 'judged_legal_pv_plies': 6,
               'first_move_no_mistake_rate': full_counts['first_move_no_mistake'] / 3,
               'full_pv_no_mistake_count': full_counts['full_pv_no_mistake']}
    for index, (row, (first, losses)) in enumerate(zip(rows, specs)):
        pv = json.loads(row['answer'])['pv']
        scored = {'row_index': index, 'id': row['id'], 'record': row, 'raw': row['answer'],
            'judgment': {'parse_valid': True, 'contract_valid': True, 'first_move_legal': True,
                'pv_legal': True, 'first_move_expected_score_loss': first,
                'first_move_no_mistake': first < .10, 'pv_move_losses': losses,
                'engine_judgments': [{'move': move} for move in pv]}}
        if mutate is not None and index == 0:
            mutate(scored)
        atomic_json(reference / 'full/rows' / f'{index:06}.json', scored)
    atomic_json(reference / 'plan.json', {'purpose': 'controlled retained-reference fixture'})
    (reference / 'readback.py').write_text('# Controlled prior reader identity fixture.\n')
    atomic_json(reference / 'full/metrics.json', metrics)
    deep = {'status': 'complete', 'kind': 'existing_training_teacher_labels_deep_engine_reference',
        'config': {'scope': 'full', 'indices': [0, 1, 2]},
        'loss_cutoff': 'expected_score_loss_strictly_less_than_0.10', 'metrics': metrics,
        'inputs': {str(p): bound(p) for p in [data / 'manifest.json', data / 'train.jsonl']},
        'outputs': {str(p): bound(p) for p in (reference / 'full/rows').glob('*.json')}}
    atomic_json(reference / 'full/manifest.json', deep)
    atomic_json(reference / 'full/exit.json', {'returncode': 0,
        'manifest_sha256': digest(reference / 'full/manifest.json')})
    readback = {'status': 'complete_training_label_reference_independent_readback', 'scope': 'full',
        'examples': 3, 'actual_retained_metric_counts': full_counts, 'recomputed_metrics': metrics,
        'producer_manifest_sha256': digest(reference / 'full/manifest.json'),
        'plan_sha256': digest(reference / 'plan.json'), 'script_sha256': digest(reference / 'readback.py'),
        'all_original_label_records_and_full_native_histories_verified': True,
        'all_stored_engine_responses_reparsed': True, 'all_complete_pv_loss_aggregates_recomputed': True,
        'original_unscored_and_same_best_additional_responses_verified': True,
        'labels_or_predictions_rewritten': False, 'validation_or_independent_test_answers_read': False}
    atomic_json(reference / 'full/independent-readback.json', readback)
    atomic_json(reference / 'full-readback-exit.json', {'returncode': 0})
    atomic_json(reference / 'full-pipeline-exit.json', {'returncode': 0,
        'independent_readback_complete': True,
        'stages': [{'stage': 'producer', 'returncode': 0}, {'stage': 'readback', 'returncode': 0}]})
    return data, reference


def test_selects_original_train_lines_and_copies_opaque_heldout_bytes_without_parsing(tmp_path):
    data, reference = fixture(tmp_path, opaque_heldout=True)
    original = (data / 'train.jsonl').read_bytes();output = tmp_path / 'qualified'
    result = qualify(data, reference, output, artifact_root=tmp_path)
    assert (output / 'train.jsonl').read_bytes() == original.splitlines(keepends=True)[0]
    for split in ['validation', 'test']:
        assert (output / f'{split}.jsonl').read_bytes() == (data / f'{split}.jsonl').read_bytes()
    assert (data / 'train.jsonl').read_bytes() == original
    assert result['by_split'] == {'train': 1, 'validation': 1, 'test': 1}
    assert result['qualification_counts']['rejected'] == 2
    assert result['validation_or_independent_test_records_parsed'] is False
    proof = checked_manifest(output / 'manifest.json')
    assert proof['kind'] == ENGINE_QUALIFIED_KIND
    source_contract(proof, result['by_split'])


def test_first_move_and_every_variation_ply_must_both_pass_the_strict_boundary(tmp_path):
    specs = [(0.099999, [0.0, 0.099999]), (0.0, [0.0, 0.10]), (0.10, [0.0, 0.0])]
    data, reference = fixture(tmp_path, specs)
    output = tmp_path / 'qualified';qualify(data, reference, output, artifact_root=tmp_path, workers=1)
    witness = json.loads((output / 'qualification.json').read_text())
    assert [row['id'] for row in witness['decisions'] if row['eligible']] == ['a']
    assert witness['decisions'][1]['first_move_passed'] is True
    assert witness['decisions'][1]['full_main_variation_passed'] is False
    assert witness['decisions'][2]['first_move_passed'] is False
    assert witness['decisions'][2]['full_main_variation_passed'] is True


@pytest.mark.parametrize('problem', ['row_id', 'record', 'first_verdict', 'missing_ply', 'nan', 'boolean_loss'])
def test_hash_bound_but_inconsistent_scored_rows_refuse_before_output(tmp_path, problem):
    def mutate(row):
        if problem == 'row_id':row['id'] = 'another-annotation'
        if problem == 'record':row['record'] = dict(copy.deepcopy(row['record']), question='changed question')
        if problem == 'first_verdict':row['judgment']['first_move_no_mistake'] = False
        if problem == 'missing_ply':row['judgment']['pv_move_losses'] = [0.0]
        if problem == 'nan':row['judgment']['first_move_expected_score_loss'] = float('nan')
        if problem == 'boolean_loss':row['judgment']['pv_move_losses'] = [True, 0.0]
    data, reference = fixture(tmp_path, mutate=mutate);output = tmp_path / 'not-created'
    with pytest.raises(ValueError):qualify(data, reference, output, artifact_root=tmp_path)
    assert not output.exists()


@pytest.mark.parametrize('problem', ['missing_readback', 'capability_only', 'changed_score_bytes', 'changed_source_bytes'])
def test_incomplete_or_changed_reference_inputs_are_not_admitted(tmp_path, problem):
    data, reference = fixture(tmp_path)
    if problem == 'missing_readback':
        p = reference / 'full-pipeline-exit.json';value = json.loads(p.read_text())
        value['independent_readback_complete'] = False;atomic_json(p, value)
    if problem == 'capability_only':
        p = reference / 'full/independent-readback.json';value = json.loads(p.read_text())
        value['scope'] = 'capability';atomic_json(p, value)
    if problem == 'changed_score_bytes':
        p = reference / 'full/rows/000000.json';p.write_bytes(p.read_bytes() + b'\n')
    if problem == 'changed_source_bytes':
        p = data / 'train.jsonl';p.write_bytes(p.read_bytes() + b'\n')
    with pytest.raises(ValueError):qualify(data, reference, tmp_path / 'not-created', artifact_root=tmp_path)
    assert not (tmp_path / 'not-created').exists()


def test_admission_rejects_changed_heldout_bytes_even_when_the_child_hash_is_updated(tmp_path):
    data, reference = fixture(tmp_path);output = tmp_path / 'qualified'
    result = qualify(data, reference, output, artifact_root=tmp_path)
    p = output / 'test.jsonl';p.write_bytes(p.read_bytes() + b' ')
    proof = json.loads((output / 'manifest.json').read_text());proof['outputs'][str(p)] = bound(p)
    atomic_json(output / 'manifest.json', proof)
    with pytest.raises(ValueError, match='held-out bytes'):
        source_contract(checked_manifest(output / 'manifest.json'), result['by_split'])


def test_existing_outputs_and_zero_eligible_corpora_are_preserved(tmp_path):
    data, reference = fixture(tmp_path, [(0.2, [0.2, 0.0])] * 3)
    with pytest.raises(ValueError, match='reference population'):
        qualify(data, reference, tmp_path / 'empty', artifact_root=tmp_path)
    assert not (tmp_path / 'empty').exists()
    output = tmp_path / 'existing';output.mkdir();marker = output / 'preserve';marker.write_text('owned prior output')
    with pytest.raises(FileExistsError):qualify(data, reference, output, artifact_root=tmp_path)
    assert marker.read_text() == 'owned prior output'


def test_existing_explanation_preflight_reaches_the_real_new_admission_seam(tmp_path, monkeypatch):
    from xqgeneral import explanation_preflight
    data, reference = fixture(tmp_path);output = tmp_path / 'qualified'
    qualify(data, reference, output, artifact_root=tmp_path)
    config = tmp_path / 'foundation.json';config.write_text('{}')
    class AdmittedBeforeCacheChecks(Exception):pass
    def admission(proof, counts):
        source_contract(proof, counts)
        raise AdmittedBeforeCacheChecks
    monkeypatch.setattr(explanation_preflight, 'clean_recipe', lambda config: None)
    monkeypatch.setattr(explanation_preflight, 'source_contract', admission)
    with pytest.raises(AdmittedBeforeCacheChecks):
        explanation_preflight.preflight(output, config, tmp_path / 'unused-cache-proof', tmp_path / 'preflight')
