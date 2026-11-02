"""Inference delivery must survive relocation and reject inconsistent trained artifacts."""
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from xqgeneral import inference, model_bundle, modeling
from xqgeneral.evidence import atomic_json, digest
from xqgeneral.feature_store import KIND as FEATURE_KIND
from xqgeneral.model_bundle import export_model, read_bundle


RAW_EVALUATION_ERROR = 'preserved malformed model output'


def bundle_evaluation(entry, training, tmp_path, monkeypatch, *, mutate=None):
    """Drive the real evaluation entry with local assets and a controlled predictor."""
    from xqgeneral import evaluate_qa, evaluate_explanations, evaluate_games, evaluate_moves, evaluate_plans
    from xqgeneral.rules import START_FEN, legal_moves, replay
    run, expert, _, _, _ = training
    bundle = tmp_path / 'evaluation-bundle'
    export_model(run, expert, bundle)
    read_bundle(bundle)
    data = tmp_path / 'evaluation-data'
    data.mkdir()
    row = dict(id='fixture-evaluation', game_id='fixture-game', split='test',
               initial_fen=START_FEN, fen=START_FEN, moves=[], history=[START_FEN],
               stage='static_current', task_type='piece', question='fixture question', answer='fixture expected')
    if entry in {'moves', 'plans'}:
        row.update(stage='move_quality' if entry == 'moves' else 'move_planning', teacher={'source': 'fixture'})
    (data / 'test.jsonl').write_text(json.dumps(row) + '\n')
    features = tmp_path / 'features.pt'
    features.write_bytes(b'fixture features')
    engine, engine_weights = tmp_path / 'engine', tmp_path / 'engine.nnue'
    engine.write_bytes(b'fixture engine'); engine_weights.write_bytes(b'fixture engine weights')
    seen = {'predictors': 0, 'generations': 0, 'experts': []}

    class ControlledPredictor:
        def __init__(self, checkpoint, weights=None, **kwargs):
            assert Path(checkpoint) == bundle
            seen['predictors'] += 1
            seen['experts'].append(weights)
            self.config = json.loads((run / 'manifest.json').read_text())['config']

        def generate_batch(self, records, *args, **kwargs):
            seen['generations'] += len(records)
            if mutate is not None:mutate(bundle)
            return [RAW_EVALUATION_ERROR] * len(records)

        def generate(self, record, **kwargs):
            return self.generate_batch([record])[0]

    class ControlledEngine:
        def __init__(self, *args, **kwargs):pass
        def choose_move(self, fen, nodes, initial_fen, moves):
            assert replay(initial_fen, moves)[-1] == fen
            return dict(best_move=legal_moves(fen)[0], requested_nodes=nodes)
        def close(self):pass

    monkeypatch.setattr(inference, 'Predictor', ControlledPredictor)
    monkeypatch.setattr(evaluate_games, 'Pikafish', ControlledEngine)
    monkeypatch.setattr(evaluate_moves, 'Pikafish', ControlledEngine)
    monkeypatch.setattr(evaluate_plans, 'Pikafish', ControlledEngine)
    monkeypatch.setattr(evaluate_games, 'OPENINGS', [('fixture-opening', [])])
    monkeypatch.setattr(evaluate_explanations, 'judge_shard', lambda rows, args:
        [dict(r, judgment=evaluate_explanations.judgment(None, r['record'], r['raw'], args.nodes)) for r in rows])
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / ('evaluated-' + entry)
    if entry == 'qa':
        argv = ['qa', '--checkpoint', str(bundle), '--data', str(data), '--features', str(features),
                '--split', 'test', '--per-task', '1', '--batch-size', '1', '--output', str(dest)]
        invoke = evaluate_qa.main
    elif entry == 'explanations':
        argv = ['explanations', '--checkpoint', str(bundle), '--data', str(data), '--features', str(features),
                '--split', 'test', '--limit', '1', '--batch-size', '1', '--workers', '1',
                '--executable', str(engine), '--weights', str(engine_weights), '--output', str(dest)]
        invoke = evaluate_explanations.main
    elif entry in {'moves', 'plans'}:
        argv = [entry, '--checkpoint', str(bundle), '--data', str(data), '--features', str(features),
                '--split', 'test', '--limit', '1', '--batch-size', '1', '--workers', '1',
                '--executable', str(engine), '--weights', str(engine_weights), '--output', str(dest)]
        invoke = evaluate_moves.main if entry == 'moves' else evaluate_plans.main
    else:
        argv = ['games', '--checkpoint', str(bundle), '--nodes', '1', '--max-plies', '1',
                '--executable', str(engine), '--weights', str(engine_weights), '--output', str(dest)]
        invoke = evaluate_games.main
    monkeypatch.setattr(sys, 'argv', argv)
    return bundle, dest, seen, invoke


@pytest.mark.parametrize('entry', ['qa', 'explanations', 'games', 'moves', 'plans'])
def test_bundle_directory_evaluation_binds_every_asset_and_preserves_raw_errors(training, tmp_path, monkeypatch, entry):
    bundle, dest, seen, invoke = bundle_evaluation(entry, training, tmp_path, monkeypatch)
    invoke()
    proof = json.loads((dest / 'manifest.json').read_text())
    assets = {str(p) for p in bundle.rglob('*') if p.is_file()}
    assert assets.issubset(proof['inputs']) and str(bundle) not in proof['inputs']
    assert all(proof['inputs'][str(p)]['sha256'] == digest(p) for p in bundle.rglob('*') if p.is_file())
    assert seen['predictors'] == 1 and seen['generations'] == 1
    if entry == 'qa':
        raw = json.loads((dest / 'predictions.jsonl').read_text())
        assert raw['generated'] == RAW_EVALUATION_ERROR and raw['correct'] is False
    elif entry in {'explanations', 'moves', 'plans'}:
        raw = json.loads((dest / 'raw-predictions.jsonl').read_text())
        assert raw['raw'] == RAW_EVALUATION_ERROR
        if entry == 'explanations':
            assert proof['verification']['parse_valid_rate'] == 0
        else:
            assert proof['verification']['format_valid_rate'] == 0
            assert proof['verification']['oracle_repairs'] == 0
    else:
        rows = [json.loads(s) for s in (dest / 'games.jsonl').read_text().splitlines()]
        assert rows[0]['turns'][0]['raw'] == RAW_EVALUATION_ERROR
        assert proof['verification']['model_oracle_repairs'] == 0
        assert Path(seen['experts'][0]) == bundle / 'expert/px0.pb.gz'


@pytest.mark.parametrize('entry', ['qa', 'explanations', 'games', 'moves', 'plans'])
def test_bundle_evaluation_rejects_corrupt_assets_before_loading_predictor(training, tmp_path, monkeypatch, entry):
    bundle, dest, seen, invoke = bundle_evaluation(entry, training, tmp_path, monkeypatch)
    (bundle / 'decoder/model.safetensors').write_bytes(b'corrupt decoder')
    with pytest.raises(ValueError, match='differs'):
        invoke()
    assert seen['predictors'] == seen['generations'] == 0
    assert not (dest / 'manifest.json').exists()


@pytest.mark.parametrize('entry', ['qa', 'explanations', 'games', 'moves', 'plans'])
def test_bundle_evaluation_rejects_assets_changed_during_generation(training, tmp_path, monkeypatch, entry):
    def mutate(bundle):
        (bundle / 'decoder/model.safetensors').write_bytes(b'changed during generation')
    _, dest, seen, invoke = bundle_evaluation(entry, training, tmp_path, monkeypatch, mutate=mutate)
    with pytest.raises(ValueError, match='changed|differs'):
        invoke()
    assert seen['predictors'] == seen['generations'] == 1
    assert not (dest / 'manifest.json').exists()


def test_bundle_match_resume_reuses_completed_games_and_rejects_rebound_asset(training, tmp_path, monkeypatch):
    from xqgeneral import evaluate_games
    bundle, dest, _, invoke = bundle_evaluation('games', training, tmp_path, monkeypatch)
    invoke()
    saved = {p: p.read_bytes() for p in dest.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):raise AssertionError('Completed games must not load either player')
    monkeypatch.setattr(inference, 'Predictor', forbidden)
    monkeypatch.setattr(evaluate_games, 'Pikafish', forbidden)
    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--resume'])
    invoke()
    assert all(p.read_bytes() == value for p, value in saved.items())
    license_path = bundle / 'decoder/LICENSE'
    license_path.write_text('changed but internally rebound license')
    path = bundle / 'manifest.json'
    proof = json.loads(path.read_text())
    proof['files']['decoder/LICENSE'] = {'sha256': digest(license_path), 'bytes': license_path.stat().st_size}
    atomic_json(path, proof)
    with pytest.raises(ValueError, match='Match resume requires'):
        invoke()
    assert all(p.read_bytes() == value for p, value in saved.items())


@pytest.mark.parametrize('entry', ['qa', 'explanations', 'games', 'moves', 'plans'])
def test_evaluation_pins_the_model_behind_a_switched_directory_alias(training, tmp_path, monkeypatch, entry):
    bundle, dest, _, invoke = bundle_evaluation(entry, training, tmp_path, monkeypatch)
    run, expert, _, _, original = training
    selected = run / 'adapter.pt'
    changed = torch.load(selected, weights_only=True)
    changed['trainable'] = {name: value + 1 for name, value in changed['trainable'].items()}
    torch.save(changed, selected)
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest['outputs'][str(selected)] = {'sha256': digest(selected), 'bytes': selected.stat().st_size}
    atomic_json(run / 'manifest.json', manifest)
    alternate = tmp_path / 'alternate-model'
    export_model(run, expert, alternate)
    alias = tmp_path / 'current-model'
    alias.symlink_to(bundle, target_is_directory=True)
    argv = list(sys.argv)
    argv[argv.index('--checkpoint') + 1] = str(alias)
    monkeypatch.setattr(sys, 'argv', argv)
    original_check = model_bundle._checked_bundle
    switched = []

    def switch_alias_after_check(path):
        result = original_check(path)
        if not switched and result[0] == bundle:
            alias.unlink()
            alias.symlink_to(alternate, target_is_directory=True)
            switched.append(True)
        return result

    class BoundPredictor:
        def __init__(self, checkpoint, weights=None, **kwargs):
            loaded, _ = read_bundle(checkpoint)
            assert torch.equal(loaded['trainable']['weight'], original['weight']), 'A different model was loaded after binding'
            self.config = loaded['config']

        def generate_batch(self, records, *args, **kwargs):
            return [RAW_EVALUATION_ERROR] * len(records)

        def generate(self, record, **kwargs):
            return RAW_EVALUATION_ERROR

    monkeypatch.setattr(model_bundle, '_checked_bundle', switch_alias_after_check)
    monkeypatch.setattr(inference, 'Predictor', BoundPredictor)
    invoke()
    assert switched and alias.resolve() == alternate
    proof = json.loads((dest / 'manifest.json').read_text())
    assert str(bundle / 'adapter.pt') in proof['inputs']
    assert str(alternate / 'adapter.pt') not in proof['inputs']


@pytest.mark.parametrize(('entry', 'decoding'), [('moves', 'raw'), ('moves', 'legal'), ('plans', 'raw')])
def test_auxiliary_evaluation_rejects_missing_batch_answers(training, tmp_path, monkeypatch, entry, decoding):
    _, dest, _, invoke = bundle_evaluation(entry, training, tmp_path, monkeypatch)
    run, _, _, _, _ = training
    argv = list(sys.argv)
    argv[argv.index('--checkpoint') + 1] = str(run / 'adapter.pt')
    argv[argv.index('--limit') + 1] = '2'
    argv[argv.index('--batch-size') + 1] = '2'
    if entry == 'moves':argv.extend(['--decoding', decoding])
    path = Path(argv[argv.index('--data') + 1]) / 'test.jsonl'
    row = json.loads(path.read_text())
    path.write_text(json.dumps(row) + '\n' + json.dumps(dict(row, id='fixture-second')) + '\n')
    requested = []

    class ShortPredictor:
        def __init__(self, checkpoint, **kwargs):
            assert Path(checkpoint).is_file()

        def generate_batch(self, rows, *args, **kwargs):
            assert len(rows) == 2
            requested[:] = [r['id'] for r in rows]
            return [RAW_EVALUATION_ERROR]

        def generate_moves(self, rows, *args, **kwargs):
            return self.generate_batch(rows)

    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(inference, 'Predictor', ShortPredictor)
    with pytest.raises(ValueError, match='coverage'):
        invoke()
    assert not (dest / 'manifest.json').exists()
    failure = json.loads((dest / 'generation-failure.json').read_text())
    assert failure['requested_ids'] == requested
    assert set(requested) == {row['id'], 'fixture-second'}
    assert failure['raw_answers'] == [RAW_EVALUATION_ERROR]
    assert failure['unpaired_outputs_not_scored'] is True
    assert not (dest / 'metrics.json').exists()


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
