from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from xqgeneral.course_tasks import PAPER_PROFILE, native_tag
from xqgeneral.curriculum_data import STAGES
from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, manifest, write_jsonl
from xqgeneral.foundation_preflight import corpus_preserves_prefix
from xqgeneral.foundation_readback import checked_course_sources
from xqgeneral.paper_curriculum import DATA_KIND, build as build_data, readback
from xqgeneral.rules import replay


def source_fixture(tmp_path):
    root = tmp_path / 'source'
    values = []
    for split, placement in [('train', 'H3K4'), ('validation', '1H2K4'), ('test', '2H1K4')]:
        for kind, prefix, last, future in [
            ('check', '4k4/9/9/9/4r4/9/9/9/9/', placement, ['e0d0']),
            ('mate-future', '4k4/3R5/5R3/9/9/9/9/9/9/', '', ['f7e7']),
        ]:
            # Keep the second probe's general on f0; its different extra horse
            # makes every split's board and color counterpart distinct.
            if kind == 'mate-future':
                last = {'train': 'H4K3', 'validation': '1H3K3', 'test': '2H2K3'}[split]
            fen = prefix + last + ' w - - 0 1'
            history = replay(fen, [])
            values.append({'game_id': split + '-' + kind + '-controlled-probe', 'split': split,
                           'initial_fen': fen, 'moves': [], 'history': history, 'fen': fen,
                           'feature_key': history_key(history), 'future_moves': future,
                           'provenance': 'constructed_native_fixture;not_human_or_engine_game'})
    roots, proof = root / 'roots.jsonl', root / 'manifest.json'
    write_jsonl(roots, values)
    atomic_json(proof, manifest('constructed_native_root_fixture', {'fixture': True}, [], [roots],
                {'model_loaded': False, 'student_training_executed': False}))
    reference = root / 'reference'
    paths = []
    for split in ('train', 'validation', 'test'):
        path = reference / f'{split}.jsonl'
        write_jsonl(path, [row for row in values if row['split'] == split])
        paths.append(path)
    atomic_json(reference / 'manifest.json', manifest('constructed_prior_corpus_fixture',
                {'fixture': True}, [], paths, {'student_training_executed': False}))
    return roots, proof, values


def build(roots, proofs, output, **kwargs):
    return build_data(roots, proofs, output, reference_data=[Path(roots[0]).parent / 'reference'], **kwargs)


def test_real_rule_generation_source_readback_and_foundation_routing_cover_all_profiles(tmp_path):
    roots, proof, originals = source_fixture(tmp_path)
    hashes = digest(roots), digest(proof)
    data = tmp_path / 'paper-data'
    result = build([roots], [proof], data, seed=7)
    assert result['questions'] == 300 and result['counts'] == {'train': 100, 'validation': 100, 'test': 100}
    assert result['source_games'] == 6 and result['all_declared_tasks_present_by_split'] == {
        'train': True, 'validation': True, 'test': True}
    assert result['paper_difficult_source_mix_applied'] is False
    assert result['feature_cache_and_token_preflight_complete'] is False
    assert result['student_training_executed'] is False
    assert hashes == (digest(roots), digest(proof))
    checked = readback(data, tmp_path / 'readback')
    assert checked['questions_regenerated_and_compared'] == 300
    produced = json.loads((data / 'manifest.json').read_text())
    assert not corpus_preserves_prefix(produced, PAPER_PROFILE)
    contexts = {}
    for split in ('train', 'validation', 'test'):
        for row in load_jsonl(data / f'{split}.jsonl'):
            assert row['game_id'] in {r['game_id'] for r in originals if r['split'] == split}
            assert row['task_profile'] == PAPER_PROFILE
            contexts.setdefault(row['feature_key'], row)
    keys, source_result, bindings = checked_course_sources(data, produced, contexts.values(),
        set(contexts), len(contexts), tmp_path / 'foundation-source-readback')
    assert keys == set(contexts) and source_result['paper_questions_regenerated'] == 300
    assert Path(bindings[0]).is_file()


@pytest.mark.parametrize('kind', ['source_changed', 'duplicate_root', 'conflicting_split', 'invalid_history'])
def test_builder_rejects_unbound_or_inconsistent_source_roots(tmp_path, kind):
    roots, proof, originals = source_fixture(tmp_path)
    if kind == 'source_changed':
        with roots.open('a') as handle:
            handle.write('{}\n')
    else:
        if kind == 'duplicate_root':
            originals.append(deepcopy(originals[0]))
        elif kind == 'conflicting_split':
            originals[2]['game_id'] = originals[0]['game_id']
        else:
            originals[0]['feature_key'] = 'wrong-native-history-key'
        write_jsonl(roots, originals)
        atomic_json(proof, manifest('constructed_native_root_fixture', {'fixture': True}, [], [roots], {}))
    with pytest.raises(ValueError):
        build([roots], [proof], tmp_path / 'invalid')


def test_game_future_and_color_leakage_are_not_accepted_as_new_task_data(tmp_path):
    roots, proof, originals = source_fixture(tmp_path)
    originals[2] = dict(originals[0], split='validation', game_id='different-game-same-board')
    write_jsonl(roots, originals)
    atomic_json(proof, manifest('constructed_native_root_fixture', {}, [], [roots], {}))
    with pytest.raises(ValueError, match='duplicate full-history'):
        build([roots], [proof], tmp_path / 'overlapping')


@pytest.mark.parametrize('change', ['label', 'source', 'extra_bytes'])
def test_readback_reconstructs_native_gold_even_if_the_modified_corpus_is_rehashed(tmp_path, change):
    roots, source_proof, originals = source_fixture(tmp_path)
    data = tmp_path / 'paper'
    build([roots], [source_proof], data, seed=7)
    proof_path = data / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    if change == 'source':
        originals[0]['future_moves'] = []
        write_jsonl(roots, originals)
        atomic_json(source_proof, manifest('constructed_native_root_fixture', {}, [], [roots], {}))
        for path in (roots, source_proof):
            proof['inputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    else:
        path = data / 'validation.jsonl'
        if change == 'label':
            rows = load_jsonl(path)
            rows[0]['answer'] = '篡改后的监督标签\n答案：无'
            write_jsonl(path, rows)
        else:
            with path.open('a') as handle:
                handle.write('\n')
        proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(proof_path, proof)
    with pytest.raises(ValueError, match='Question bytes|extra rows'):
        readback(data, tmp_path / 'reject-readback')


def test_legacy_prefix_and_new_profile_cannot_be_silently_interchanged(tmp_path):
    roots, proof, _ = source_fixture(tmp_path)
    data = tmp_path / 'paper'
    build([roots], [proof], data)
    produced = json.loads((data / 'manifest.json').read_text())
    assert produced['kind'] == DATA_KIND
    with pytest.raises(ValueError, match='preserved base-data'):
        corpus_preserves_prefix(produced, 'legacy')
    with pytest.raises(ValueError, match='declared native question'):
        corpus_preserves_prefix(dict(produced, kind='legacy_producer'), PAPER_PROFILE)
    with pytest.raises(ValueError, match='fresh output'):
        build([roots], [proof], data)
    unchecked = tmp_path / 'no-prior-isolation'
    build_data([roots], [proof], unchecked)
    produced = json.loads((unchecked / 'manifest.json').read_text())
    assert produced['verification']['prior_corpus_isolation_verified'] is False
    with pytest.raises(ValueError, match='declared native question'):
        corpus_preserves_prefix(produced, PAPER_PROFILE)


def test_prior_heldout_future_cannot_enter_the_new_training_pool(tmp_path):
    roots, proof, originals = source_fixture(tmp_path)
    reference = roots.parent / 'reference'
    # Retain only the old training root, while its one-move child is a prior
    # validation position. The proposed new line must therefore be rejected.
    from xqgeneral.rules import play
    previous = [dict(originals[0], future_moves=[]),
                dict(originals[2], future_moves=[]), dict(originals[4], future_moves=[])]
    target = play(originals[0]['fen'], originals[0]['future_moves'][0])
    previous[1].update(initial_fen=target, fen=target, history=[target], moves=[],
                       feature_key=history_key([target]), game_id='old-heldout-child')
    paths = []
    for row in previous:
        path = reference / f"{row['split']}.jsonl"
        write_jsonl(path, [row])
        paths.append(path)
    atomic_json(reference / 'manifest.json', manifest('constructed_prior_corpus_fixture', {}, [], paths, {}))
    with pytest.raises(ValueError, match='prior corpus split'):
        build([roots], [proof], tmp_path / 'leaked-future')


def test_raw_qa_cli_uses_tag_grading_and_keeps_unmodified_generation(tmp_path, monkeypatch):
    from xqgeneral import evaluate_qa, inference
    roots, proof, _ = source_fixture(tmp_path)
    data = tmp_path / 'paper'
    build([roots], [proof], data)
    checkpoint, features = tmp_path / 'checkpoint.pt', tmp_path / 'features.pt'
    checkpoint.write_bytes(b'controlled-inference-placeholder;not-a-model')
    features.write_bytes(b'controlled-cache-placeholder')

    class ControlledPredictor:
        config = {'task_profile': PAPER_PROFILE}

        def __init__(self, *args, **kwargs):
            pass

        def generate_batch(self, batch, *args):
            return ['另一种解释。\n答案：' + native_tag(row) for row in batch]

    monkeypatch.setattr(inference, 'Predictor', ControlledPredictor)
    output = tmp_path / 'qa'
    monkeypatch.setattr(sys, 'argv', ['qa', '--checkpoint', str(checkpoint), '--data', str(data),
        '--features', str(features), '--per-task', '1', '--question-formats', '3', '--stages', *STAGES,
        '--output', str(output)])
    evaluate_qa.main()
    records = load_jsonl(output / 'predictions.jsonl')
    assert len(records) == 26 and all(r['generated'].startswith('另一种解释。') for r in records)
    metrics = json.loads((output / 'metrics.json').read_text())
    assert metrics['accuracy'] == 1 and metrics['raw_exact_correct_rate'] == 0
    assert metrics['prose_graded'] is False
    assert json.loads((output / 'manifest.json').read_text())['config']['task_profile'] == PAPER_PROFILE
