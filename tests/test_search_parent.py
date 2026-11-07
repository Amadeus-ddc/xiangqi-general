"""Native toy completion/raw/certificate consumers; no real neural or engine claims."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import test_auxiliary_policy_evaluation as auxiliary_control
import test_clean_sft_evaluation as paired_control
import test_oracle_policy_corpus as corpus_control
from xqgeneral import clean_sft_evaluation as evaluation, search_parent as admission, search_protection
from xqgeneral.evidence import atomic_json, code_identity, digest, load_jsonl, write_jsonl
from xqgeneral.recorded_search_inputs import BoundInputs


def artifact(path):
    return {'sha256': digest(path), 'bytes': Path(path).stat().st_size}


def case(tmp_path, monkeypatch):
    # The source pool and pre-initial model are controlled upstream seams. The
    # actual seven-stage corpus, three-worker completion consumer, paired raw
    # validation and certificate reconstruction below run their native code.
    original_fixture = corpus_control.fixture
    def completed_authority(path, **kwargs):
        args = original_fixture(path, **kwargs)
        contract_path = args['policies'] / 'contract.json'
        contract = json.loads(contract_path.read_text()); shared = contract['shared_isolation_authority']
        authority = Path(shared['authority']) / 'manifest.json'
        atomic_json(authority, dict(status='complete', kind='prepared_unused_recorded_search_roots',
                                   controlled_upstream_source=True))
        shared['authority_manifest_binding'] = artifact(authority); atomic_json(contract_path, contract)
        proof_path = args['policies'] / 'manifest.json'; proof = json.loads(proof_path.read_text())
        proof['outputs'][str(contract_path)] = artifact(contract_path); atomic_json(proof_path, proof)
        return args
    monkeypatch.setattr(corpus_control, 'fixture', completed_authority)
    pipeline, partial, recipe, result = auxiliary_control.native_completed_auxiliary(
        tmp_path, monkeypatch, world=3, validation_histories=2, parent_rounds=0)
    token = json.loads(Path(partial['token_preflight']).read_text())
    data = Path(token['config']['logical']['data'])
    corpus_contract = json.loads((data / 'contract.json').read_text())
    initial = token['config']['logical']['source']; completion = token['config']['logical']['parent_completion']
    parent = json.loads((pipeline / 'manifest.json').read_text())
    native_parent = admission.completed_parent
    def previous(source, path, bound):
        if str(source) == initial:
            assert str(path) == completion
            bound.bind(path, parent['inputs'][str(path)])
            bound.bind(source, parent['inputs'][str(source)])
            return {}, dict(kind=admission.INITIAL_ROUND_KIND,
                outputs={initial: parent['inputs'][initial]}, verification={})
        return native_parent(source, path, bound)
    monkeypatch.setattr(admission, 'completed_parent', previous)
    teacher_root = tmp_path / 'teacher'; teacher_root.mkdir()
    shard = teacher_root / 'model.safetensors'; shard.write_bytes(b'controlled BF16 identity only')
    teacher = dict(repository='Qwen/Qwen3.8-27B', revision='controlled-fixture-only', quantization='none',
                   inference_dtype='bfloat16', model_path=str(teacher_root))
    atomic_json(teacher_root / 'weights.manifest.json', dict(teacher, all_official_lfs_sha256_matched=True,
        stored_tensor_dtypes={'BF16': 1}, total_weight_bytes=shard.stat().st_size,
        shards={'model.safetensors': artifact(shard)}))
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        (teacher_root / name).write_text('{}')
    teacher_path = tmp_path / 'teacher.json'; atomic_json(teacher_path, {'search_consolidator': teacher})
    config = json.loads(Path('configs/evaluation-clean-sft-v1.json').read_text())
    config.update(partial, data=corpus_contract['explanations'][0], teacher_config=str(teacher_path),
        validation_examples=2, batch_size=2, executable=str(tmp_path / 'engine'), weights=str(tmp_path / 'nnue'), workers=1)
    config_path = tmp_path / 'evaluation.json'; atomic_json(config_path, config)
    controlled = paired_control.ControlledEvaluation(subprocess.run, monkeypatch)
    output = tmp_path / 'paired'; evaluation.run_evaluation(pipeline, 'controlled-owned-round', config_path, output)
    source, native = evaluation.checked_sft(pipeline, config, recipe)
    certificate = tmp_path / 'protection'; search_protection.prepare(data, certificate)
    capability_path = output / 'manifest.json'; capability = json.loads(capability_path.read_text())
    settings = dict(data=next(iter(corpus_contract['original_shared_isolation_authority']['base_data_identities'])))
    settings['data'] = str(Path(settings['data']).parent)
    settings['protection_manifest'] = str(certificate / 'manifest.json')
    pool = corpus_contract['original_shared_isolation_authority']['authority']
    return source, native, token, capability_path, capability, config, settings, pool, controlled


def admit(value):
    return admission.admit_descendant(*value[:8], BoundInputs())


def test_native_three_worker_parent_paired_errors_and_full_certificate_reopen_without_generation(tmp_path, monkeypatch):
    value = case(tmp_path, monkeypatch)
    before = len(value[-1].commands)
    note = admit(value)
    assert note['completed_ancestor_counts_rechecked'] and note['every_original_paired_raw_answer_and_metric_reopened']
    assert note['search_protection']['all_raw_reference_variations_and_mirrors_reconstructed']
    assert note['search_protection']['corpus'] == str(tmp_path / 'out')
    assert note['full_blinded_review_raw_responses_and_rejections_reopened']
    assert not note['independent_engine_rejudgment_performed'] and not note['raw_answers_repaired']
    assert len(value[-1].commands) == before
    assert value[4]['verification']['trained_auxiliary_policy_rounds'] == 1
    assert value[4]['verification']['trained_search_distillation_rounds'] == 0
    assert all(m['parse_valid_rate'] == 0 for m in value[4]['verification']['raw_validation_by_memory'].values())


@pytest.mark.parametrize('damage', ['missing_certificate', 'wrong_corpus', 'wrong_family', 'search_count',
    'auxiliary_count', 'initial_flag', 'missing_memory', 'raw_header_only', 'raw_repaired', 'raw_source'])
def test_unqualified_descendants_cannot_reenter_search(tmp_path, monkeypatch, damage):
    value = list(case(tmp_path, monkeypatch))
    capability, settings = value[4], value[6]
    if damage == 'missing_certificate': settings.pop('protection_manifest')
    elif damage == 'wrong_corpus':
        path = Path(settings['protection_manifest']); p = json.loads(path.read_text())
        p['config']['corpus'] = str(tmp_path / 'unrelated-corpus'); atomic_json(path, p)
    elif damage == 'wrong_family': value[1] = dict(value[1], kind=admission.ROUND_KIND)
    elif damage == 'search_count': capability['verification']['trained_search_distillation_rounds'] = 1
    elif damage == 'auxiliary_count': capability['verification']['trained_auxiliary_policy_rounds'] = 0
    elif damage == 'initial_flag': capability['verification']['actual_clean_initial_sft_completed_before_model_or_reviewer_load'] = True
    else:
        memory = Path(value[3]).parent / 'shuffled'; manifest_path = memory / 'manifest.json'
        p = json.loads(manifest_path.read_text())
        if damage == 'missing_memory': capability['verification']['raw_validation_by_memory'].pop('shuffled')
        elif damage == 'raw_header_only': capability['inputs'].pop(str(manifest_path))
        elif damage == 'raw_repaired':
            path = memory / 'judged-predictions.jsonl'; rows = load_jsonl(path)
            rows[0]['raw'] = 'a repaired answer'; write_jsonl(path, rows)
            p['outputs'][str(path)] = artifact(path); capability['outputs'][str(path)] = artifact(path)
        else: p['code']['revision'] = 'unrelated-neural-source'
        if damage in ['raw_repaired', 'raw_source']:
            atomic_json(manifest_path, p); capability['inputs'][str(manifest_path)] = artifact(manifest_path)
    with pytest.raises(ValueError):
        admit(value)


def frozen_raw_source(value, tmp_path):
    root = tmp_path / 'preserved-raw-source'; source = root / 'source/xqgeneral'; source.mkdir(parents=True)
    current = Path(admission.__file__).parent
    for path in current.glob('*.py'): shutil.copyfile(path, source / path.name)
    declared = dict(revision='controlled-frozen-raw-v1', source_sha256={p.name: digest(p) for p in source.glob('*.py')})
    path = root / 'source-manifest.json'; atomic_json(path, declared)
    expected = dict(revision=declared['revision'], source_sha256={'src/xqgeneral/' + k: v for k, v in declared['source_sha256'].items()})
    for memory in ['normal', 'zero', 'shuffled']:
        proof_path = Path(value[3]).parent / memory / 'manifest.json'; p = json.loads(proof_path.read_text())
        p['code'] = expected; atomic_json(proof_path, p); value[4]['inputs'][str(proof_path)] = artifact(proof_path)
    for stage in ['queries', 'reviewer', 'ratings']:
        proof_path = Path(value[3]).parent / stage / 'manifest.json'; p = json.loads(proof_path.read_text())
        p['code'] = expected
        for name in p['inputs']:
            if Path(name).name == 'manifest.json': p['inputs'][name] = artifact(Path(name))
        atomic_json(proof_path, p); value[4]['inputs'][str(proof_path)] = artifact(proof_path)
    value[6]['validation_source_manifest'] = dict(path=str(path), sha256=digest(path))
    return path


def test_explicit_frozen_raw_inventory_can_replay_without_claiming_inference_equivalence(tmp_path, monkeypatch):
    value = list(case(tmp_path, monkeypatch)); frozen_raw_source(value, tmp_path)
    note = admit(value)
    assert note['historical_raw_source_explicitly_declared']
    assert not note['different_inference_deployments_claimed_equivalent']
    value[6].pop('validation_source_manifest')
    with pytest.raises(ValueError, match='Raw evaluation differs'): admit(value)


@pytest.mark.parametrize('damage', ['manifest', 'file', 'dependency', 'metric_checker'])
def test_historical_source_declarations_cannot_hide_changed_replay_contracts(tmp_path, monkeypatch, damage):
    value = list(case(tmp_path, monkeypatch)); path = frozen_raw_source(value, tmp_path)
    if damage == 'manifest': value[6]['validation_source_manifest']['sha256'] = '0' * 64
    else:
        name = 'clean_sft_evaluation.py' if damage == 'metric_checker' else 'rules.py'
        file = path.parent / 'source/xqgeneral' / name
        text = file.read_text()
        if damage == 'metric_checker': text = text.replace('def checked_raw(', 'def changed_checked_raw(', 1)
        else: text += '\n# changed preserved dependency\n'
        file.write_text(text)
        if damage != 'file':
            p = json.loads(path.read_text()); p['source_sha256'][name] = digest(file); atomic_json(path, p)
            value[6]['validation_source_manifest']['sha256'] = digest(path)
    with pytest.raises(ValueError): admit(value)


@pytest.mark.parametrize('damage', ['missing_stage', 'missing_response', 'rating_repaired', 'false_summary'])
def test_complete_raw_memories_cannot_hide_missing_or_changed_blinded_review(tmp_path, monkeypatch, damage):
    value = list(case(tmp_path, monkeypatch)); capability = value[4]; root = Path(value[3]).parent
    if damage == 'missing_stage': capability['inputs'].pop(str(root / 'ratings/manifest.json'))
    else:
        if damage == 'missing_response':
            directory = root / 'reviewer'; path = directory / 'responses.jsonl'
            write_jsonl(path, load_jsonl(path)[1:])
        else:
            directory = root / 'ratings'; path = directory / ('ratings.jsonl' if damage == 'rating_repaired' else 'manifest.json')
            if damage == 'rating_repaired':
                rows = load_jsonl(path); rows[0]['clarity'] = 5; write_jsonl(path, rows)
        manifest_path = directory / 'manifest.json'; p = json.loads(manifest_path.read_text())
        if damage == 'false_summary':
            p['verification']['valid_ratings'] += 1; capability['verification']['blinded_prose_review'] = p['verification']
        else:
            p['outputs'][str(path)] = artifact(path); capability['outputs'][str(path)] = artifact(path)
        atomic_json(manifest_path, p); capability['inputs'][str(manifest_path)] = artifact(manifest_path)
    with pytest.raises(ValueError): admit(value)


def lineage(tmp_path, monkeypatch):
    initial = tmp_path / 'initial.pt'; initial.write_text('controlled initial')
    initial_completion = tmp_path / 'initial.json'
    original = dict(kind=admission.INITIAL_ROUND_KIND, outputs={str(initial): artifact(initial)}, verification={})
    atomic_json(initial_completion, original)
    proofs = {str(initial): original}; current_source, completion = initial, initial_completion
    for i, kind in enumerate([admission.AUXILIARY_ROUND_KIND, admission.ROUND_KIND], 1):
        source = tmp_path / ('round' + str(i) + '.pt'); source.write_text('controlled round')
        token_path = tmp_path / ('token' + str(i) + '.json')
        logical = dict(source=str(current_source), parent_completion=str(completion), data=str(tmp_path / 'corpus'))
        token = dict(kind=admission.AUXILIARY_TOKEN_KIND if i == 1 else admission.TOKEN_KIND,
                     config=dict(logical=logical) if i == 1 else logical)
        atomic_json(token_path, token)
        p = dict(kind=kind, config=dict(source=str(current_source), token_preflight=str(token_path),
            token_preflight_sha256=digest(token_path)), inputs={str(token_path): artifact(token_path),
            str(current_source): artifact(current_source), str(completion): artifact(completion)},
            outputs={str(source): artifact(source)}, verification=dict(trained_auxiliary_policy_rounds=1,
            trained_search_distillation_rounds=i - 1))
        proofs[str(source)] = p; completion = tmp_path / ('complete' + str(i) + '.json'); atomic_json(completion, p)
        current_source = source
    def controlled(source, path, bound):
        proof = bound.json(path); assert proof == proofs[str(source)]
        bound.bind(source, proof['outputs'][str(source)]); return {}, proof
    monkeypatch.setattr(admission, 'completed_parent', controlled)
    return current_source, p, token


def test_search_descendants_preserve_auxiliary_footprint_and_both_native_counts(tmp_path, monkeypatch):
    source, proof, token = lineage(tmp_path, monkeypatch)
    assert admission.auxiliary_ancestors(source, proof, token, BoundInputs()) == {str(tmp_path / 'corpus')}
    assert admission.round_counts(proof) == dict(trained_auxiliary_policy_rounds=1, trained_search_distillation_rounds=1)


@pytest.mark.parametrize('key,value', [('trained_auxiliary_policy_rounds', 0), ('trained_search_distillation_rounds', 2),
                                     ('trained_search_distillation_rounds', True)])
def test_search_lineage_cannot_lose_auxiliary_or_invent_search_rounds(tmp_path, monkeypatch, key, value):
    source, proof, token = lineage(tmp_path, monkeypatch); proof = deepcopy(proof); proof['verification'][key] = value
    with pytest.raises(ValueError): admission.auxiliary_ancestors(source, proof, token, BoundInputs())
