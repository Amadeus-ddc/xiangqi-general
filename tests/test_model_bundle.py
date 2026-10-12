"""Inference delivery must survive relocation and reject inconsistent trained artifacts."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xqgeneral import inference, modeling
from xqgeneral.evidence import atomic_json, digest
from xqgeneral.feature_store import KIND as FEATURE_KIND
from xqgeneral.model_bundle import export_model, read_bundle


@pytest.fixture
def training(tmp_path):
    base = tmp_path / 'original-decoder'
    base.mkdir()
    for name, content in {'config.json': '{"model_type":"qwen3"}', 'tokenizer_config.json': '{}',
                          'tokenizer.json': '{}', 'model.safetensors': 'fixture weights',
                          'LICENSE': 'fixture license'}.items():
        (base / name).write_text(content)
    hidden = base / '.cache'
    hidden.mkdir()
    (hidden / 'download-lock').write_text('local download metadata')
    (base / 'download-state.json').write_text('local download state')
    expert = tmp_path / 'original-px0.pb.gz'
    expert.write_bytes(b'fixture expert identity')
    cache = tmp_path / 'original-cache'
    cache.mkdir()
    atomic_json(cache / 'manifest.json', {'kind': FEATURE_KIND, 'status': 'complete',
        'code': {'source_sha256': {'src/xqgeneral/expert.py': digest(Path(inference.__file__).with_name('expert.py'))}},
        'verification': {'depths': [0], 'expert_sha256': digest(expert)}})
    run = tmp_path / 'training'
    run.mkdir()
    config = {'mode': 'bridge', 'model_path': str(base), 'feature_path': str(cache / 'manifest.json'),
              'data_path': str(tmp_path / 'private-training-data'), 'decoder_training': 'frozen',
              'expert_feature_depths': [0], 'decoder_bridge_positions': [0], 'bridge_width': 2,
              'board_tokens': False, 'trainable_parameter_dtype': 'float32', 'model_revision': 'a' * 40,
              'mixture': {'move_quality': 1.0}}
    code = {'revision': 'fixture', 'source_sha256': {}}
    state = {'weight': torch.tensor([[0.5, 0.25], [0.125, 0.0625]]), 'bias': torch.tensor([0.25, 0.5])}
    torch.save({'config': config, 'trainable': state, 'selected_step': 2, 'code': code}, run / 'adapter.pt')
    def binding(path):
        return {'sha256': digest(path), 'bytes': path.stat().st_size}
    proof = {'kind': 'model_training', 'status': 'complete', 'config': config, 'code': code,
             'inputs': {str(cache / 'manifest.json'): binding(cache / 'manifest.json')},
             'outputs': {str(run / 'adapter.pt'): binding(run / 'adapter.pt')},
             'verification': {'selected_step': 2, 'steps_completed': 3, 'trainable_parameters': 6}}
    atomic_json(run / 'manifest.json', proof)
    return run, expert, base, cache, state


def test_relocated_bundle_loads_exact_parameters_without_original_paths(training, tmp_path, monkeypatch):
    run, expert, base, cache, state = training
    bundle = tmp_path / 'bundle'
    proof = export_model(run, expert, bundle)
    moved = tmp_path / 'relocated'
    bundle.rename(moved)
    for path in (run, base, cache):
        path.rename(path.with_name(path.name + '-unavailable'))
    expert.rename(expert.with_name('unavailable-expert'))
    captured = {}
    def local_decoder(config, device):
        captured.update(config)
        assert Path(config['model_path']) == moved / 'decoder'
        assert (Path(config['model_path']) / 'model.safetensors').read_text() == 'fixture weights'
        return nn.Linear(2, 2), object()
    monkeypatch.setattr(modeling, 'load_model', local_decoder)
    model, _, config = modeling.load_checkpoint(moved, device='cpu')
    assert not model.training
    assert all(torch.equal(dict(model.named_parameters())[name], value) for name, value in state.items())
    assert 'data_path' not in captured and 'feature_path' not in captured
    assert config['inference_expert_path'] == str(moved / 'expert/px0.pb.gz')
    assert proof['verification']['quality_assessed_by_export'] is False
    assert not (moved / 'decoder/.cache').exists()
    assert not (moved / 'decoder/download-state.json').exists()
    assert (moved / 'decoder/LICENSE').read_text() == 'fixture license'


def test_export_accepts_feature_store_directory_reference(training, tmp_path):
    run, expert, _, cache, _ = training
    checkpoint = torch.load(run / 'adapter.pt', weights_only=True)
    checkpoint['config']['feature_path'] = str(cache)
    torch.save(checkpoint, run / 'adapter.pt')
    proof = json.loads((run / 'manifest.json').read_text())
    proof['config'] = checkpoint['config']
    proof['outputs'][str(run / 'adapter.pt')] = {'sha256': digest(run / 'adapter.pt'), 'bytes': (run / 'adapter.pt').stat().st_size}
    atomic_json(run / 'manifest.json', proof)
    export_model(run, expert, tmp_path / 'bundle')
    read_bundle(tmp_path / 'bundle')


def test_full_decoder_export_preserves_base_and_bridge_parameters(training, tmp_path, monkeypatch):
    run, expert, _, _, state = training
    class CompleteModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2, 2)
            self.bridges = nn.ModuleList([nn.Linear(2, 2)])
    full = {**{'base.' + n: t for n, t in state.items()},
            **{'bridges.0.' + n: t + 1 for n, t in state.items()}}
    checkpoint = torch.load(run / 'adapter.pt', weights_only=True)
    checkpoint['trainable'] = full
    checkpoint['config']['decoder_training'] = 'full'
    torch.save(checkpoint, run / 'adapter.pt')
    proof = json.loads((run / 'manifest.json').read_text())
    proof['config'] = checkpoint['config']
    proof['verification']['trainable_parameters'] = 12
    proof['outputs'][str(run / 'adapter.pt')] = {'sha256': digest(run / 'adapter.pt'), 'bytes': (run / 'adapter.pt').stat().st_size}
    atomic_json(run / 'manifest.json', proof)
    export_model(run, expert, tmp_path / 'bundle')
    monkeypatch.setattr(modeling, 'load_model', lambda config, device: (CompleteModel(), None))
    model, _, config = modeling.load_checkpoint(tmp_path / 'bundle', device='cpu')
    assert config['decoder_training'] == 'full'
    assert set(dict(model.named_parameters())) == set(full)
    assert all(torch.equal(dict(model.named_parameters())[n], t) for n, t in full.items())


def test_bundle_predictor_uses_bound_expert_and_rejects_wrong_override(training, tmp_path, monkeypatch):
    run, expert, _, _, _ = training
    bundle = tmp_path / 'bundle'
    export_model(run, expert, bundle)
    checkpoint, _ = read_bundle(bundle)
    monkeypatch.setattr(inference, 'load_checkpoint', lambda path, device: (nn.Linear(2, 2), None, checkpoint['config']))
    seen = []
    def local_expert(path):
        seen.append(Path(path))
        return nn.Identity()
    monkeypatch.setattr(inference, 'FrozenPx0', local_expert)
    inference.Predictor(bundle, device='cpu')
    assert seen == [bundle / 'expert/px0.pb.gz']
    wrong = tmp_path / 'wrong-expert'
    wrong.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='exported trained expert'):
        inference.Predictor(bundle, weights=wrong, device='cpu')
    assert len(seen) == 1


@pytest.mark.parametrize('change', ['incomplete', 'wrong_kind', 'wrong_selection', 'wrong_count', 'unbound_checkpoint',
                                   'unbound_cache', 'wrong_depths', 'wrong_encoder', 'wrong_expert', 'zero_selection'])
def test_rejects_unverified_training_or_expert_before_export(training, tmp_path, change):
    run, expert, base, cache, _ = training
    path = run / 'manifest.json'
    proof = json.loads(path.read_text())
    if change == 'incomplete': proof['status'] = 'running'
    elif change == 'wrong_kind': proof['kind'] = 'preflight_only'
    elif change == 'wrong_selection': proof['verification']['selected_step'] = 1
    elif change == 'wrong_count': proof['verification']['trainable_parameters'] = 5
    elif change == 'unbound_checkpoint': proof['outputs'] = {}
    elif change == 'unbound_cache': proof['inputs'] = {}
    elif change == 'wrong_expert': expert.write_bytes(b'another expert')
    elif change == 'zero_selection':
        checkpoint = torch.load(run / 'adapter.pt', weights_only=True)
        checkpoint['selected_step'] = 0
        torch.save(checkpoint, run / 'adapter.pt')
        proof['outputs'][str(run / 'adapter.pt')] = {'sha256': digest(run / 'adapter.pt'), 'bytes': (run / 'adapter.pt').stat().st_size}
        proof['verification']['selected_step'] = 0
    else:
        feature = json.loads((cache / 'manifest.json').read_text())
        if change == 'wrong_depths': feature['verification']['depths'] = [1]
        if change == 'wrong_encoder': feature['code']['source_sha256']['src/xqgeneral/expert.py'] = '0' * 64
        atomic_json(cache / 'manifest.json', feature)
        proof['inputs'][str(cache / 'manifest.json')] = {'sha256': digest(cache / 'manifest.json'), 'bytes': (cache / 'manifest.json').stat().st_size}
    atomic_json(path, proof)
    with pytest.raises(ValueError):
        export_model(run, expert, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()
    assert (base / 'model.safetensors').exists()


@pytest.mark.parametrize('change', ['corrupt_expert', 'corrupt_decoder', 'extra_weights', 'symlink_expert',
                                   'path_escape', 'incompatible_runtime'])
def test_rejects_changed_or_unbound_bundle_assets(training, tmp_path, change):
    run, expert, _, _, _ = training
    bundle = tmp_path / 'bundle'
    export_model(run, expert, bundle)
    if change == 'corrupt_expert': (bundle / 'expert/px0.pb.gz').write_bytes(b'corrupt')
    elif change == 'corrupt_decoder': (bundle / 'decoder/model.safetensors').write_bytes(b'corrupt')
    elif change == 'extra_weights': (bundle / 'decoder/pytorch_model.bin').write_bytes(b'unbound override')
    elif change == 'symlink_expert':
        (bundle / 'expert/px0.pb.gz').unlink()
        (bundle / 'expert/px0.pb.gz').symlink_to(expert)
    elif change == 'path_escape':
        proof = json.loads((bundle / 'manifest.json').read_text())
        proof['files']['../original-px0.pb.gz'] = proof['files']['expert/px0.pb.gz']
        atomic_json(bundle / 'manifest.json', proof)
    else:
        proof = json.loads((bundle / 'manifest.json').read_text())
        proof['code']['source_sha256']['src/xqgeneral/expert.py'] = '0' * 64
        atomic_json(bundle / 'manifest.json', proof)
    with pytest.raises(ValueError):
        read_bundle(bundle)


def test_existing_output_and_missing_decoder_shard_preserve_sources(training, tmp_path):
    run, expert, base, _, _ = training
    output = tmp_path / 'exists'
    output.mkdir()
    (output / 'user-file').write_text('keep')
    with pytest.raises(ValueError, match='Preserve previous'):
        export_model(run, expert, output)
    assert (output / 'user-file').read_text() == 'keep'
    atomic_json(base / 'model.safetensors.index.json', {'weight_map': {'weight': 'missing.safetensors'}})
    with pytest.raises(ValueError, match='missing or external shards'):
        export_model(run, expert, tmp_path / 'missing-shard')
    assert not (tmp_path / 'missing-shard').exists()
