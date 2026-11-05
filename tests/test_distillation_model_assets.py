"""Real native rows with data-only foundation receipts and separately pinned assets."""
import json
from pathlib import Path

import pytest

from test_distillation_preflight import cpu_flow
from test_distillation_training import budget_token, round_setup, template
from xqgeneral import distillation_preflight, distillation_training, sft_preflight
from xqgeneral.evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from test_sft_pipeline import setup_data


def pinned(model, revision, path):
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']:
        if not (model / name).exists():
            atomic_json(model / name, {})
    proof = manifest('pinned_local_decoder_assets',
        {'model_path': str(model), 'model_revision': revision}, sorted(model.iterdir()), [], {})
    atomic_json(path, proof)
    return {'path': str(path), 'sha256': digest(path)}, proof['inputs']


def native_pinned_flow(tmp_path, monkeypatch):
    source, parent, data, cache, foundation, _, tokenized = cpu_flow(tmp_path, monkeypatch)
    model = tmp_path / 'model'
    reference, assets = pinned(model, 'controlled-fixed-revision', tmp_path / 'decoder-pin.json')
    atomic_json(foundation, manifest('recorded_engine_clean_latent_foundation_preflight', {}, [], [], {}))
    previous = distillation_preflight.completed_parent
    def with_real_pin(path, completion, bound):
        saved, proof = previous(path, completion, bound)
        return dict(saved, model_revision='controlled-fixed-revision', pinned_base_model_manifest=reference), proof
    monkeypatch.setattr(distillation_preflight, 'completed_parent', with_real_pin)
    return (source, parent, data, cache, foundation), model, reference, assets, tokenized


def test_native_preflight_uses_actual_parent_pin_with_a_data_only_foundation(tmp_path, monkeypatch):
    args, _, reference, assets, tokenized = native_pinned_flow(tmp_path, monkeypatch)
    output = tmp_path / 'native-token-proof'
    result = distillation_preflight.preflight(*args, output, workers=1)
    proof = json.loads((output / 'manifest.json').read_text())
    assert result['pinned_base_model_manifest'] == reference
    assert proof['config']['pinned_base_model_manifest'] == reference
    assert proof['inputs'][reference['path']]['sha256'] == reference['sha256']
    assert all(proof['inputs'][name] == binding for name, binding in assets.items())
    assert len(tokenized) == 34 and not result['model_weights_loaded_or_training_started']


def test_changed_pinned_weights_cannot_reach_student_tokenization(tmp_path, monkeypatch):
    args, model, _, _, tokenized = native_pinned_flow(tmp_path, monkeypatch)
    (model / 'model.safetensors').write_bytes(b'changed after pinning')
    output = tmp_path / 'must-not-complete'
    with pytest.raises(ValueError, match='completed recorded search source'):
        distillation_preflight.preflight(*args, output, workers=1)
    assert not tokenized and not (output / 'manifest.json').exists()


def token_with_pin():
    token = budget_token(1000)
    reference = {'path': 'actual-decoder-pin.json', 'sha256': 'a' * 64}
    token['config']['pinned_base_model_manifest'] = reference
    token['inputs'] = {reference['path']: {'sha256': reference['sha256'], 'bytes': 123}}
    return token, reference


def test_compiler_carries_the_exact_token_verified_pin_into_gpu_and_training_recipe(tmp_path):
    token, reference = token_with_pin()
    recipe, _ = distillation_training.compile_recipe(template(), token, tmp_path)
    assert recipe['pinned_base_model_manifest'] == reference


@pytest.mark.parametrize('problem', ['override', 'missing_binding', 'invalid_digest'])
def test_unverified_or_replaced_decoder_pin_cannot_compile(tmp_path, problem):
    token, reference = token_with_pin(); recipe = template()
    if problem == 'override': recipe['pinned_base_model_manifest'] = dict(reference, sha256='b' * 64)
    elif problem == 'missing_binding': token['inputs'] = {}
    else: token['config']['pinned_base_model_manifest'] = dict(reference, sha256='invalid')
    with pytest.raises(ValueError, match='pinned decoder manifest'):
        distillation_training.compile_recipe(recipe, token, tmp_path)


def test_complete_round_propagates_pin_with_no_decoder_weights_in_data_foundation(tmp_path, monkeypatch):
    source, recipe, token_path, runner, _ = round_setup(tmp_path, monkeypatch)
    config = json.loads((source.parent / 'config.json').read_text())
    reference, _ = pinned(Path(config['model_path']), config.get('model_revision'), tmp_path / 'round-decoder-pin.json')
    foundation = tmp_path / 'new-data-only-foundation.json'
    atomic_json(foundation, manifest('recorded_engine_clean_latent_foundation_preflight', {}, [], [], {}))
    token = json.loads(token_path.read_text())
    token['config'].update(foundation_preflight=str(foundation), pinned_base_model_manifest=reference)
    for path in [foundation, Path(reference['path'])]:
        token['inputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(token_path, token)
    root = tmp_path / 'round'
    result = distillation_training.run_round(source, recipe, token_path, root)
    compiled = json.loads((root / 'recipe.json').read_text())
    saved = json.loads((root / 'sft/config.json').read_text())
    assert compiled['pinned_base_model_manifest'] == saved['pinned_base_model_manifest'] == reference
    assert result['actual_distillation_sft_completed'] and len(runner.jobs) == 4


def mixed_length_probe(tmp_path, *, longest_old=130, observed_id='old-train'):
    _, recipe_path, token_path = setup_data(tmp_path)
    recipe = json.loads(recipe_path.read_text()); token = json.loads(token_path.read_text())
    replay = tmp_path / 'replay'
    lengths = next(Path(path) for path in token['outputs'] if path.endswith('token-lengths.jsonl'))
    observations = load_jsonl(lengths)
    for split, size in [('train', longest_old), ('validation', 114)]:
        row = {'id': 'old-' + split, 'stage': 'static_current', 'split': split, 'answer': 'old answer'}
        write_jsonl(replay / (split + '.jsonl'), [row])
        observations.append({'id': observed_id if split == 'train' else row['id'], 'split': split, 'tokens': size})
    write_jsonl(lengths, observations)
    recipe.update(replay_data_paths=[str(replay)], stages=['explanation', 'static_current'],
                  mixture={'explanation': .96, 'static_current': .04})
    return recipe, token


def test_mixed_length_probe_selects_primary_rows_and_covers_shorter_replay_capacity(tmp_path):
    recipe, token = mixed_length_probe(tmp_path)
    result = sft_preflight.probe_data(recipe, token, tmp_path / 'probe')
    assert result['train']['maximum_tokens'] == 131
    assert result['train']['replay_maximum_tokens'] == 130
    assert len(result['train']['ids']) == 24
    assert all(identity.startswith('train') for identity in result['train']['ids'])


def test_mixed_length_probe_refuses_replay_larger_than_native_probe_capacity(tmp_path):
    recipe, token = mixed_length_probe(tmp_path, longest_old=200)
    with pytest.raises(ValueError, match='Replay sequences exceed'):
        sft_preflight.probe_data(recipe, token, tmp_path / 'probe')


def test_mixed_length_probe_refuses_unknown_observation_ids(tmp_path):
    recipe, token = mixed_length_probe(tmp_path, observed_id='unbound-example')
    with pytest.raises(ValueError, match='actual replay sources'):
        sft_preflight.probe_data(recipe, token, tmp_path / 'probe')
