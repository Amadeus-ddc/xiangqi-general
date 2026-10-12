"""CPU contracts use controlled producer receipts, never real mined labels."""
import json
from pathlib import Path
import random

import pytest
import torch

from test_feature_store import block, prepared
from test_search_collection import query, response, TEACHER
from xqgeneral import collect_search, distillation_preflight
from xqgeneral.evidence import atomic_json, digest, history_key, manifest, write_jsonl
from xqgeneral.feature_store import FeatureStoreWriter, feature_paths
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, legal_moves, play


def extended(tmp_path):
    _, old, original, _ = prepared(tmp_path, width=512)
    writer = FeatureStoreWriter(tmp_path / 'extended', [0, 3], 'a' * 64,
        config={'base_cache': str(old), 'base_proof': str(old)})
    writer.add_existing(old, old)
    writer.append(block(['new-history'], width=512))
    writer.finish()
    return old, writer.output / 'manifest.json', original


def test_extended_cache_guard_preserves_original_tensor_references_and_bytes(tmp_path):
    old, new, original = extended(tmp_path)
    before = {str(p): digest(p) for p in feature_paths(old)}
    bound = BoundInputs()
    cache = distillation_preflight.extended_cache(new, {'feature_path': str(old),
        'expert_feature_depths': [0, 3]}, bound)
    assert cache.keys == [*original.keys, 'new-history']
    assert cache.shards[:-1] == original.shards
    for actual, previous in zip(cache.features, original.features):
        assert torch.equal(actual[:len(original.keys)], previous[:])
    assert torch.equal(cache.wdl[:len(original.keys)], original.wdl[:])
    assert all(digest(p) == value for p, value in before.items())
    assert set(map(str, feature_paths(old) + feature_paths(new))) <= bound.artifacts.keys()


@pytest.mark.parametrize('problem', ['same_cache', 'missing_parent', 'foreign_parent', 'expert',
    'depth', 'corrupt_tensor', 'reordered_keys'])
def test_extended_cache_guard_refuses_wrong_lineage_or_changed_bytes(tmp_path, problem):
    old, new, original = extended(tmp_path)
    saved = {'feature_path': str(old), 'expert_feature_depths': [0, 3]}
    proof = json.loads(new.read_text())
    if problem == 'same_cache': new = old
    elif problem == 'missing_parent':
        proof['config'].pop('base_cache'); atomic_json(new, proof)
    elif problem == 'foreign_parent':
        proof['config']['base_cache'] = str(tmp_path / 'absent-parent.json'); atomic_json(new, proof)
    elif problem == 'expert':
        proof['verification']['expert_sha256'] = 'b' * 64; atomic_json(new, proof)
    elif problem == 'depth': saved['expert_feature_depths'] = [0, 4]
    elif problem == 'corrupt_tensor':
        path = Path(original.shards[0]['path']); path.write_bytes(path.read_bytes() + b'changed')
    else:
        path = Path(proof['catalog']['keys']); keys = json.loads(path.read_text())
        keys[0], keys[1] = keys[1], keys[0]; atomic_json(path, keys)
        proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        atomic_json(new, proof)
    with pytest.raises(ValueError):
        distillation_preflight.extended_cache(new, saved, BoundInputs())


def collected(tmp_path):
    source = tmp_path / 'selected-parent.pt'; source.write_bytes(b'controlled parent binding')
    queries = tmp_path / 'mining/queries.jsonl'; queries.parent.mkdir()
    queries.write_text('controlled query bytes\n')
    mining_path = queries.parent / 'manifest.json'
    atomic_json(mining_path, manifest('search_distillation_mining', {'checkpoint': str(source)},
        [source], [queries], {'unused_recorded_training_inputs': True}))
    original, data = tmp_path / 'protected', tmp_path / 'labels'
    original.mkdir(); data.mkdir()
    (data / 'train.jsonl').write_text('controlled training bytes\n')
    inputs, outputs = [mining_path, queries], [data / 'train.jsonl']
    for split in ['validation', 'test']:
        path = original / (split + '.jsonl'); path.write_text(split + ' preserved bytes\n')
        copied = data / path.name; copied.write_bytes(path.read_bytes())
        inputs.append(path); outputs.append(copied)
    result = dict.fromkeys(['neural_consolidation_executed', 'structured_facts_verified',
        'explicit_prose_moves_natively_verified', 'all_branch_positions_isolated',
        'full_history_termination_checked', 'recorded_mining_isolation_rechecked_during_collection',
        'selected_branch_evaluation_filter_applied'], True)
    result.update(accepted_training_roots=32, validation_source_unchanged=str(original),
        evaluation_filter_profile=collect_search.EVALUATION_FILTER_PROFILE,
        max_absolute_winrate_error_exclusive=.075,
        teacher={'quantization': 'none', 'inference_dtype': 'bfloat16'})
    atomic_json(data / 'manifest.json', manifest('search_consolidated_dataset',
        {'queries': str(queries), 'mining_manifest': str(mining_path)}, inputs, outputs, result))
    return data, source, queries, original


def test_collected_contract_rehashes_miner_and_preserves_protected_bytes(tmp_path):
    data, source, queries, original = collected(tmp_path)
    bound = BoundInputs()
    proof = distillation_preflight.collected_dataset(data, source, bound)
    assert proof['verification']['accepted_training_roots'] == 32
    assert str(source) in bound.artifacts and str(queries) in bound.artifacts
    for split in ['validation', 'test']:
        assert (original / (split + '.jsonl')).read_bytes() == (data / (split + '.jsonl')).read_bytes()


@pytest.mark.parametrize('problem', ['pending', 'roots_only', 'tiny', 'no_native', 'no_filter',
    'loose_filter', 'quantized', 'wrong_source', 'changed_query', 'changed_validation', 'changed_parent'])
def test_collected_contract_rejects_unfinished_or_stale_producers(tmp_path, problem):
    data, source, queries, original = collected(tmp_path)
    path = data / 'manifest.json'; proof = json.loads(path.read_text())
    if problem == 'pending': proof['status'] = 'pending'
    elif problem == 'roots_only': proof['kind'] = 'prepared_unused_recorded_search_roots'
    elif problem == 'tiny': proof['verification']['accepted_training_roots'] = 31
    elif problem == 'no_native': proof['verification']['structured_facts_verified'] = False
    elif problem == 'no_filter': proof['verification'].pop('selected_branch_evaluation_filter_applied')
    elif problem == 'loose_filter': proof['verification']['max_absolute_winrate_error_exclusive'] = .1
    elif problem == 'quantized': proof['verification']['teacher']['quantization'] = 'AWQ'
    elif problem == 'wrong_source':
        mining = queries.parent / 'manifest.json'; value = json.loads(mining.read_text())
        value['config']['checkpoint'] = 'other-parent.pt'; atomic_json(mining, value)
        proof['inputs'][str(mining)] = {'sha256': digest(mining), 'bytes': mining.stat().st_size}
    elif problem == 'changed_query': queries.write_text('changed controlled query\n')
    elif problem == 'changed_validation': (data / 'validation.jsonl').write_text('changed labels\n')
    else: source.write_bytes(b'changed parent')
    atomic_json(path, proof)
    with pytest.raises(ValueError):
        distillation_preflight.collected_dataset(data, source, BoundInputs())


@pytest.mark.parametrize('error,accepted', [(0, True), (.0745, True), (.075, False), (.0755, False), (1, False)])
def test_exclusive_error_boundary_does_not_disappear_after_oracle_rewrite(monkeypatch, error, accepted):
    q = query()
    branch = dict(q['target_fields']['branches'][0]['evaluation'])
    q['target_fields']['branches'][0]['evaluation'] = branch
    def probability(fen, evaluation):
        return .5 + error if evaluation is branch else .5
    monkeypatch.setattr(collect_search, 'evaluation_probability', probability)
    if accepted:
        value = collect_search.search_evaluation_filter(q)
        assert value['absolute_winrate_error'] == pytest.approx(error)
        assert not value['consolidated_prose_evaluation_error_measured']
    else:
        with pytest.raises(ValueError, match='exclusive packing threshold'):
            collect_search.search_evaluation_filter(q)


@pytest.mark.parametrize('threshold', [0, -1, 1.01, True, float('nan'), float('inf')])
def test_invalid_evaluation_filter_budget_is_rejected(threshold):
    with pytest.raises(ValueError):
        collect_search.search_evaluation_filter(query(), threshold)


def cpu_flow(tmp_path, monkeypatch):
    """Native synthetic boards, fake producer parent and tokenizer; CPU only."""
    data, source, _, original = collected(tmp_path)
    rows = []
    for number in range(35):
        rng = random.Random(1200 + number)
        fen, moves, history = START_FEN, [], [START_FEN]
        for _ in range(12):
            move = rng.choice(legal_moves(fen)); fen = play(fen, move)
            moves.append(move); history.append(fen)
        q = query(); q['id'] = f'controlled-{number}'
        q['record'].update(initial_fen=START_FEN, fen=fen, moves=moves, history=history,
            game_id=f'controlled-game-{number}', feature_key=history_key(history))
        candidates = list(legal_moves(fen)[:2]); score = q['target_fields']['evaluation']
        q['target_fields'].update(move=candidates[0], pv=[candidates[0]], candidates=candidates,
            facts=move_facts(fen, candidates[0]),
            branches=[{'move': m, 'pv': [m], 'evaluation': score} for m in candidates])
        raw = response(); raw['id'] = q['id']
        row = collect_search.consolidated_label(q, raw, TEACHER)
        row['split'] = 'train' if number < 32 else ('validation' if number < 34 else 'test')
        rows.append(row)
    proof = json.loads((data / 'manifest.json').read_text())
    for split in ['train', 'validation', 'test']:
        path = data / (split + '.jsonl'); write_jsonl(path, [r for r in rows if r['split'] == split])
        proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        if split != 'train':
            protected = original / path.name; protected.write_bytes(path.read_bytes())
            proof['inputs'][str(protected)] = proof['outputs'][str(path)]
    atomic_json(data / 'manifest.json', proof)
    base = FeatureStoreWriter(tmp_path / 'base-features', [0, 3], 'a' * 64)
    base.append(block([r['feature_key'] for r in rows], width=512)); base.finish()
    previous = base.output / 'manifest.json'
    writer = FeatureStoreWriter(tmp_path / 'new-features', [0, 3], 'a' * 64,
        config={'base_cache': str(previous)})
    writer.add_existing(previous, previous)
    writer.append(block(['unused-extra-history'], width=512)); writer.finish()
    model = tmp_path / 'model'; model.mkdir()
    weights = model / 'model.safetensors'; weights.write_bytes(b'controlled base identity')
    foundation = tmp_path / 'foundation.json'
    atomic_json(foundation, manifest('recorded_engine_clean_latent_foundation_preflight', {}, [weights]))
    parent = tmp_path / 'controlled-parent.json'; atomic_json(parent, {'controlled': True})
    saved = {'feature_path': str(previous), 'expert_feature_depths': [0, 3], 'model_path': str(model)}
    def controlled_parent(path, completion, bound):
        bound.bind(path); bound.bind(completion)
        return saved, {'controlled_parent': True}
    monkeypatch.setattr(distillation_preflight, 'completed_parent', controlled_parent)
    tokenized = []
    monkeypatch.setattr(distillation_preflight, 'token_initializer', lambda *args: None)
    def tokenize(batch):
        tokenized.extend(r['id'] for r in batch)
        return [{'id': r['id'], 'split': r['split'], 'tokens': 240, 'supervised_tokens': 200,
                 'prompt_tokens': 40} for r in batch]
    monkeypatch.setattr(distillation_preflight, 'token_chunk', tokenize)
    return source, parent, data, writer.output / 'manifest.json', foundation, rows, tokenized


def test_cpu_flow_checks_all_native_rows_and_never_tokenizes_test(tmp_path, monkeypatch):
    source, parent, data, cache, foundation, rows, tokenized = cpu_flow(tmp_path, monkeypatch)
    output = tmp_path / 'cpu-preflight'
    result = distillation_preflight.preflight(source, parent, data, cache, foundation, output, workers=1)
    assert result['by_split'] == {'train': 32, 'validation': 2, 'test': 1}
    assert result['original_selected_branch_evaluation_filter_rechecked']
    assert result['independent_test_rows_tokenized'] == 0
    assert tokenized == [r['id'] for r in rows if r['split'] != 'test']
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['kind'] == distillation_preflight.TOKEN_KIND
    assert str(data / 'test.jsonl') in proof['inputs']
    assert not result['model_weights_loaded_or_training_started']


def test_cpu_token_failure_keeps_partial_output_without_completion(tmp_path, monkeypatch):
    source, parent, data, cache, foundation, _, _ = cpu_flow(tmp_path, monkeypatch)
    def too_long(batch):
        raise ValueError('supervised answer exceeds token limit without truncation')
    monkeypatch.setattr(distillation_preflight, 'token_chunk', too_long)
    output = tmp_path / 'cpu-preflight'
    with pytest.raises(ValueError, match='without truncation'):
        distillation_preflight.preflight(source, parent, data, cache, foundation, output, workers=1)
    assert (output / 'token-lengths.jsonl').exists()
    assert not (output / 'manifest.json').exists()
