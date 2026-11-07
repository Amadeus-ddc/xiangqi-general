"""Real CPU Adam states and finite indexes; no neural inference or GPU execution."""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import pytest
import torch

import test_auxiliary_policy_preflight as input_control
from xqgeneral import auxiliary_policy_preflight as auxiliary
from xqgeneral import auxiliary_resource_transition as transition
from xqgeneral import auxiliary_policy_training as round_training
from xqgeneral.distillation_preflight import completed_parent
from xqgeneral.evidence import atomic_json, code_identity, digest, manifest
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.sft_preflight import identical
from xqgeneral.training import select_validation_rows, training_input_paths, training_optimizer, training_memory_settings
from xqgeneral.training import atomic_checkpoint
from xqgeneral.training_index import IndexedTrainingRows, build_index


def template(world):
    name = 'three' if world == 3 else 'two'
    value = json.loads(Path('configs/auxiliary-policy-sft-' + name + '-gpu-v1.json').read_text())
    value['eval_every'] = 2
    return value


def original_state(tmp_path, monkeypatch, *, build_target=True, micro=1):
    args, saved, _ = input_control.fixture(tmp_path, monkeypatch)
    saved['decoder_training'] = 'full'
    names = ['base.model.embed_tokens.board_weight', 'base.decoder.weight', 'bridges.0.weight']
    reference = {name: torch.zeros(2, 3) for name in names}
    torch.save(dict(config=saved, trainable=reference), args['source'])
    atomic_json(args['source'].parent / 'config.json', saved)
    auxiliary.prepare(**args)
    token_path = args['output'] / 'manifest.json'; token = json.loads(token_path.read_text())
    old = tmp_path / 'original-round'; old.mkdir()
    recipe_path = tmp_path / 'original-template.json'; atomic_json(recipe_path, template(3))
    recipe, config, budget = auxiliary.compile_recipe(token, template(3), old)
    source_code = code_identity()
    contract = dict(source=str(args['source']), source_sha256=digest(args['source']),
        recipe_template=str(recipe_path), recipe_template_sha256=digest(recipe_path),
        token_preflight=str(token_path), token_preflight_sha256=digest(token_path),
        compiled_recipe=recipe, code=source_code)
    atomic_json(old / 'contract.json', contract); atomic_json(old / 'recipe.json', recipe)
    native = old / 'sft'; native.mkdir(); atomic_json(native / 'config.json', config)
    for split in ['train', 'validation']:
        build_index([args['data'] / (split + '.jsonl')], recipe['row_index_paths'][split], split=split)
    rows = IndexedTrainingRows(recipe['row_index_paths']['train'], [args['data'] / 'train.jsonl'], recipe['mixture'])
    validation = IndexedTrainingRows(recipe['row_index_paths']['validation'],
        [args['data'] / 'validation.jsonl'], recipe['mixture'], split='validation')
    try:
        finite = prepare_finite_epochs(rows, recipe['mixture'], config, 3)
        ids = [r['id'] for r in select_validation_rows(validation, config['validation_stages'], config['seed'], config['validation_examples'])]
        atomic_json(native / 'validation_ids.json', ids)
        parameters = {name: torch.nn.Parameter(value.clone()) for name, value in reference.items()}
        model = SimpleNamespace(named_parameters=lambda: iter(parameters.items()))
        optimizer = training_optimizer(model, config)
        logs = []
        for step in [1, 2]:
            for index, parameter in enumerate(parameters.values()):
                parameter.grad = torch.full_like(parameter, (step + index) / 10)
            norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            logs.append(dict(step=step, loss=1 / step, gradient_norm=float(norm), supervised_tokens=step * 10,
                lr=optimizer.param_groups[0]['lr'], seconds=float(step), data_batch=finite.batch_indices(step - 1, 0)[1]))
        prefix = ''.join(json.dumps(row) + '\n' for row in logs).encode()
        (native / 'training.jsonl').write_bytes(prefix)
        hashes = {str(path): digest(path) for path in training_input_paths(config)}
        latest = dict(trainable={name: p.detach().clone() for name, p in parameters.items()},
            optimizer=optimizer.state_dict(), step=2, world_size=3, config=config, best_step=2,
            best_loss=.5, initial_loss=1., patience_count=0, tokens_seen=20,
            random_state=random.Random(7).getstate(), torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=[torch.tensor([i, i + 1], dtype=torch.uint8) for i in range(3)],
            input_hashes=hashes, code=source_code, compute_contract=dict(deterministic_training=True,
                cublas_workspace_config=':4096:8', **training_memory_settings(config)),
            data_state=finite.state_at(2))
    finally:
        rows.close(); validation.close()
    selected = {k: deepcopy(latest[k]) for k in ['trainable', 'config', 'input_hashes', 'code']}
    selected['selected_step'] = 2
    torch.save(latest, native / 'latest.pt'); torch.save(selected, native / 'adapter.pt')
    release = tmp_path / 'release'; preserved = release / 'preserved-checkpoints'; preserved.mkdir(parents=True)
    for name in ['latest.pt', 'adapter.pt']:
        shutil.copyfile(native / name, preserved / name)
    atomic_json(release / 'preserved-checkpoints.json', dict(saved_step=2,
        files={name: dict(path=str(preserved / name)) for name in ['latest.pt', 'adapter.pt']}))
    worker = lambda i: dict(pid=-10000 - i, start_time=1, command_sha256='0' * 64,
        rank_environment=dict(RANK=str(i), WORLD_SIZE='3', CUDA_VISIBLE_DEVICES='0,1,3'))
    atomic_json(release / 'release-plan.json', dict(training_output=str(native), world_size=3,
        launcher=worker(10), GPU_workers=[worker(i) for i in range(3)]))
    atomic_json(release / 'release-verification.json', dict(project_GPU_workers_all_terminal=True,
        parent_attempt_interrupted_for_user_resource_request=True, other_existing_GPU_occupants_not_signalled=True,
        durable_native_checkpoint_step=2))
    frozen = tmp_path / 'frozen-source'; package = frozen / 'source/xqgeneral'; package.mkdir(parents=True)
    source_dir = Path(transition.__file__).parent
    for path in source_dir.glob('*.py'):
        shutil.copyfile(path, package / path.name)
    freeze = frozen / 'source-freeze.json'
    atomic_json(freeze, dict(source_path=str(package.parent),
        source_sha256={p.name: digest(p) for p in package.glob('*.py')}))
    target = tmp_path / 'target-round'
    target_template = template(2)
    if micro == 3:
        target_template.update(auxiliary_topology_profile='full_decoder_auxiliary_two_workers_micro3_v1', micro_batch_size=3)
    compiled, target_config, _ = auxiliary.compile_recipe(token, target_template, target)
    if build_target:
        for split in ['train', 'validation']:
            build_index([args['data'] / (split + '.jsonl')], compiled['row_index_paths'][split], split=split)
    return dict(round_directory=old, release_directory=release, source_freeze=freeze,
        capture=tmp_path / 'source-capture', bundle=tmp_path / 'bundle', config=target_config,
        latest=latest, selected=selected, native=native, prefix=prefix, target=target,
        token=token_path, source=args['source'], saved=saved, target_recipe=compiled)


def capture(value):
    return transition.capture_source(value['round_directory'], value['release_directory'],
                                     value['source_freeze'], value['capture'])


@pytest.mark.parametrize('micro', [1, 3])
def test_cpu_partial_state_preserves_real_Adam_and_every_parameter_across_two_workers(tmp_path, monkeypatch, micro):
    value = original_state(tmp_path, monkeypatch, micro=micro)
    source = capture(value)
    assert source['saved_step'] == 2 and source['optimizer']['parameter_states'] == 3
    result = transition.prepare_transition(value['capture'] / 'manifest.json', value['config'],
        value['bundle'], retained_ranks=[0, 2])
    state = torch.load(value['bundle'] / 'latest.pt', weights_only=True)
    assert state['world_size'] == 2 and state['step'] == 2
    for field in ['trainable', 'optimizer', 'random_state', 'torch_rng_state', 'tokens_seen', 'patience_count', 'best_loss', 'best_step']:
        assert identical(state[field], value['latest'][field])
    assert identical(state['cuda_rng_states'], [value['latest']['cuda_rng_states'][0], value['latest']['cuda_rng_states'][2]])
    assert result['consumed_global_batches_and_next_valid_row_order_compared'] == 3
    assert result['first_pass_unique_rows_consumed'] == 60 and not result['exact_future_three_worker_trajectory_claimed']
    assert result['source_micro_batch_size'] == 1 and result['target_micro_batch_size'] == micro
    assert transition.checked_transition(value['bundle']) == result


@pytest.mark.parametrize('damage', ['release', 'partial_count', 'nonfinite_parameter', 'moments', 'missing_Adam',
    'selection', 'cursor', 'log_prefix', 'tokens', 'validation', 'rng'])
def test_invalid_original_native_state_never_creates_a_transition_source(tmp_path, monkeypatch, damage):
    value = original_state(tmp_path, monkeypatch)
    path = value['release_directory'] / 'preserved-checkpoints/latest.pt'
    saved = torch.load(path, weights_only=True)
    if damage == 'release':
        p = value['release_directory'] / 'release-verification.json'; proof = json.loads(p.read_text())
        proof['project_GPU_workers_all_terminal'] = False; atomic_json(p, proof)
    elif damage == 'log_prefix': (value['native'] / 'training.jsonl').write_bytes(value['prefix'].splitlines(keepends=True)[1])
    elif damage == 'validation': atomic_json(value['native'] / 'validation_ids.json', ['omitted original rows'])
    else:
        if damage == 'partial_count': saved['step'] = 0
        elif damage == 'nonfinite_parameter': next(iter(saved['trainable'].values())).fill_(float('inf'))
        elif damage == 'moments': saved['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
        elif damage == 'missing_Adam': saved['optimizer']['state'].pop(0)
        elif damage == 'selection': saved['best_step'] = 3
        elif damage == 'cursor': saved['data_state']['optimizer_batch_in_epoch'] += 1
        elif damage == 'tokens': saved['tokens_seen'] += 1
        else: saved['cuda_rng_states'].pop()
        torch.save(saved, path)
    with pytest.raises(ValueError): capture(value)
    assert not value['capture'].exists()


@pytest.mark.parametrize('change', [dict(seed=7), dict(learning_rate=2e-5), dict(batch_size=32),
    dict(max_tokens=1024), dict(steps=49), dict(ddp_world_size=3)])
def test_transition_cannot_change_training_data_geometry_or_schedule(tmp_path, monkeypatch, change):
    value = original_state(tmp_path, monkeypatch); capture(value)
    with pytest.raises(ValueError):
        transition.prepare_transition(value['capture'] / 'manifest.json', dict(value['config'], **change),
                                      value['bundle'], retained_ranks=[0, 2])
    assert not value['bundle'].exists()


def test_native_installation_keeps_bundle_immutable_when_training_log_appends(tmp_path, monkeypatch):
    value = original_state(tmp_path, monkeypatch); capture(value)
    result = transition.prepare_transition(value['capture'] / 'manifest.json', value['config'],
        value['bundle'], retained_ranks=[0, 2])
    native = Path(value['config']['output'])
    assert transition.install_transition(value['bundle'], native) == result
    before = (value['bundle'] / 'training-prefix.jsonl').read_bytes()
    with (native / 'training.jsonl').open('ab') as stream:
        stream.write(b'{"step": 3, "controlled_CPU_update": true}\n')
    assert (value['bundle'] / 'training-prefix.jsonl').read_bytes() == before == value['prefix']
    assert transition.install_transition(value['bundle'], native) == result
    assert (native / 'training.jsonl').read_bytes().endswith(b'"controlled_CPU_update": true}\n')


def test_complete_capture_and_bundle_resume_preserve_all_output_bytes_and_mtimes(tmp_path, monkeypatch):
    value = original_state(tmp_path, monkeypatch); capture(value)
    result = transition.prepare_transition(value['capture'] / 'manifest.json', value['config'],
        value['bundle'], retained_ranks=[0, 2])
    paths = [*value['capture'].iterdir(), *value['bundle'].iterdir()]
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    transition.capture_source(value['round_directory'], value['release_directory'], value['source_freeze'], value['capture'], resume=True)
    assert transition.prepare_transition(value['capture'] / 'manifest.json', value['config'],
        value['bundle'], retained_ranks=[0, 2], resume=True) == result
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}


def test_native_atomic_checkpoint_replacement_preserves_initial_transition_bundle(tmp_path, monkeypatch):
    value = original_state(tmp_path, monkeypatch); capture(value)
    result = transition.prepare_transition(value['capture'] / 'manifest.json', value['config'],
        value['bundle'], retained_ranks=[0, 2])
    native = Path(value['config']['output']); transition.install_transition(value['bundle'], native)
    initial = (value['bundle'] / 'latest.pt').read_bytes()
    later = torch.load(native / 'latest.pt', weights_only=True); later['step'] = 3
    atomic_checkpoint(native / 'latest.pt', later)
    assert (value['bundle'] / 'latest.pt').read_bytes() == initial
    assert transition.checked_transition(value['bundle']) == result
    transition.install_transition(value['bundle'], native)
    assert torch.load(native / 'latest.pt', weights_only=True)['step'] == 3


def completed_resource_round(tmp_path, monkeypatch, *, micro=1, damage=None):
    value = original_state(tmp_path, monkeypatch, build_target=False, micro=micro); capture(value)
    target_template = template(2)
    if micro == 3:
        target_template.update(auxiliary_topology_profile='full_decoder_auxiliary_two_workers_micro3_v1', micro_batch_size=3)
    path = tmp_path / 'target-template.json'; atomic_json(path, target_template)
    calls = []
    def parent(source, completion, bound):
        return auxiliary.completed_parent(source, completion, bound)[0], dict(verification=dict(
            trained_auxiliary_policy_rounds=0, trained_search_distillation_rounds=2))
    monkeypatch.setattr(round_training, 'completed_parent', parent)
    def probe(source, recipe, token, output, resume=False):
        calls.append('GPU_preflight'); output = Path(output); output.mkdir()
        initial = output / 'zero-init.pt'
        torch.save(dict(trainable=torch.load(source, weights_only=True)['trainable']), initial)
        result = dict(status='complete', world_size=2, trainable_tensors=3,
            actual_continuous_four_and_two_plus_two_updates_compared=True,
            full_decoder_initialization_preserved_all_parent_tensors=True)
        proof = manifest('two_gpu_auxiliary_policy_sft_preflight',
            dict(source=str(source), recipe=str(recipe), token_preflight=str(token)), [], [initial], result)
        proof['verified_feature_cache_input'] = json.loads(Path(token).read_text())['cache_header_only_input']
        atomic_json(output / 'manifest.json', proof)
        return result
    monkeypatch.setattr(round_training, 'run_preflight', probe)
    def formal(command, **kwargs):
        calls.append('formal_SFT'); native = Path(command[command.index('--output') + 1])
        saved = torch.load(native / 'latest.pt', weights_only=True)
        assert saved['step'] == 2 and saved['world_size'] == 2
        assert identical(saved['optimizer'], value['latest']['optimizer'])
        assert (native / 'training.jsonl').read_bytes() == value['prefix']
        config = saved['config']; weights = {n: torch.nn.Parameter(t.clone()) for n, t in saved['trainable'].items()}
        optimizer = training_optimizer(SimpleNamespace(named_parameters=lambda: iter(weights.items())), config)
        optimizer.load_state_dict(saved['optimizer'])
        rows = IndexedTrainingRows(config['row_index_paths']['train'],
            [Path(config['data_path']) / 'train.jsonl'], config['mixture'])
        try:
            finite = prepare_finite_epochs(rows, config['mixture'], config, 2)
            with (native / 'training.jsonl').open('ab') as log:
                for step in range(3, config['steps'] + 1):
                    for parameter in weights.values(): parameter.grad = torch.full_like(parameter, .1)
                    norm = torch.nn.utils.clip_grad_norm_(list(weights.values()), 1.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    item = dict(step=step, loss=1 / step, gradient_norm=float(norm), supervised_tokens=step * 10,
                        lr=optimizer.param_groups[0]['lr'], seconds=float(step), data_batch=finite.batch_indices(step - 1, 0)[1])
                    log.write((json.dumps(item) + '\n').encode())
            state = dict(saved, trainable={n: p.detach().clone() for n, p in weights.items()},
                optimizer=optimizer.state_dict(), step=config['steps'], best_step=config['steps'], best_loss=.1,
                tokens_seen=config['steps'] * 10, data_state=finite.state_at(config['steps']))
        finally: rows.close()
        atomic_checkpoint(native / 'latest.pt', state)
        selected = {k: state[k] for k in ['trainable', 'config', 'code', 'input_hashes']}
        selected['selected_step'] = state['step']; atomic_checkpoint(native / 'adapter.pt', selected)
        metrics = dict(world_size=2, decoder_training='full', global_batch_size=30, selected_step=state['step'],
            steps_completed=state['step'], training_data_state=state['data_state'], trainable_parameters=18)
        execution = dict(code=code_identity(), input_hashes=state['input_hashes'], resumed_from_step=2,
                         compute_contract=state['compute_contract'])
        if damage == 'resumed_step': execution['resumed_from_step'] = 0
        elif damage == 'input_hash': execution['input_hashes'] = {}
        elif damage == 'marker': (native / 'resource-transition-installed.json').unlink()
        atomic_json(native / 'metrics.json', metrics); atomic_json(native / 'execution.json', execution)
        proof = manifest('model_training', config, training_input_paths(config),
            [native / name for name in ['adapter.pt', 'config.json', 'metrics.json', 'validation_ids.json', 'execution.json']], metrics)
        atomic_json(native / 'manifest.json', proof)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(round_training, 'subprocess', SimpleNamespace(run=formal, STDOUT=-2))
    arguments = dict(source=value['source'], recipe_template=path, token_manifest=value['token'], output=value['target'],
        resource_source=value['capture'] / 'manifest.json', retained_ranks=[0, 2])
    return arguments, value, calls


@pytest.mark.parametrize('micro', [1, 3])
def test_native_two_worker_round_continues_preserved_updates_and_completes_one_auxiliary_round(tmp_path, monkeypatch, micro):
    arguments, value, calls = completed_resource_round(tmp_path, monkeypatch, micro=micro)
    result = round_training.run_round(**arguments)
    assert calls == ['GPU_preflight', 'formal_SFT']
    ledger = result['training_resource_transition']
    assert ledger['original_three_worker_updates'] == 2 and ledger['new_two_worker_updates'] == 46
    assert ledger['total_global_updates'] == 48 and ledger['target_micro_batch_size'] == micro
    assert result['trained_auxiliary_policy_rounds'] == 1 and result['trained_search_distillation_rounds'] == 2
    saved, proof = completed_parent(Path(result['selected_checkpoint']), value['target'] / 'manifest.json')
    assert saved['ddp_world_size'] == 2 and proof['verification'] == result
    assert round_training.run_round(**arguments, resume=True) == result
    assert calls == ['GPU_preflight', 'formal_SFT']
    assert transition.checked_transition(value['target'] / 'resource-transition')['source_saved_step'] == 2


@pytest.mark.parametrize('damage', ['resumed_step', 'input_hash', 'marker'])
def test_native_training_cannot_claim_the_wrong_resource_resume_boundary(tmp_path, monkeypatch, damage):
    arguments, value, calls = completed_resource_round(tmp_path, monkeypatch, damage=damage)
    with pytest.raises((ValueError, FileNotFoundError)):
        round_training.run_round(**arguments)
    assert not (value['target'] / 'manifest.json').exists()
    assert calls == ['GPU_preflight', 'formal_SFT']


@pytest.mark.parametrize('damage', ['reference', 'total_updates', 'strip_ledger'])
def test_completed_parent_rechecks_resource_transition_provenance(tmp_path, monkeypatch, damage):
    arguments, value, _ = completed_resource_round(tmp_path, monkeypatch)
    result = round_training.run_round(**arguments)
    path = value['target'] / 'manifest.json'; proof = json.loads(path.read_text())
    if damage == 'reference': proof['config']['resource_source']['saved_step'] = 1
    elif damage == 'total_updates': proof['verification']['training_resource_transition']['total_global_updates'] += 1
    else: proof['verification'].pop('training_resource_transition')
    atomic_json(path, proof)
    with pytest.raises(ValueError): completed_parent(Path(result['selected_checkpoint']), path)


def test_microbatch_three_candidate_uses_native_two_worker_capacity_and_exact_resume_gate(tmp_path, monkeypatch):
    import test_auxiliary_three_workers as controls
    value = template(2); value.update(auxiliary_topology_profile='full_decoder_auxiliary_two_workers_micro3_v1', micro_batch_size=3)
    monkeypatch.setattr(controls, 'template', lambda: value)
    root, result, _, _, _, runner = controls.three_worker_probe(tmp_path, monkeypatch)
    assert result['world_size'] == 2 and result['exact_resume']['mismatched_fields'] == []
    assert result['exact_resume']['world_sizes'] == [2, 2]
    assert json.loads((root / 'continuous/config.json').read_text())['micro_batch_size'] == 3
    assert len(result['actual_gpu_worker_resources']) == 6 and len(runner.jobs) == 3


@pytest.mark.parametrize('filename,allowed', [('training.py', False), ('search_pilot.py', True)])
def test_original_code_inventory_allows_reviewed_metadata_changes_and_rejects_changed_training_math(tmp_path, monkeypatch,
                                                                                               filename, allowed):
    value = original_state(tmp_path, monkeypatch)
    frozen = json.loads(value['source_freeze'].read_text()); path = Path(frozen['source_path']) / 'xqgeneral' / filename
    path.write_text(path.read_text() + '\n# controlled preserved-source difference\n')
    frozen['source_sha256'][filename] = digest(path); atomic_json(value['source_freeze'], frozen)
    code = deepcopy(value['latest']['code']); code['source_sha256']['src/xqgeneral/' + filename] = digest(path)
    for name in ['latest.pt', 'adapter.pt']:
        p = value['release_directory'] / 'preserved-checkpoints' / name
        state = torch.load(p, weights_only=True); state['code'] = code; torch.save(state, p)
    p = value['round_directory'] / 'contract.json'; contract = json.loads(p.read_text()); contract['code'] = code; atomic_json(p, contract)
    if allowed:
        result = capture(value)
        assert result['source_code_transition']['changed_metadata_modules'] == [filename]
    else:
        with pytest.raises(ValueError, match='Training computation changed'): capture(value)
        assert not value['capture'].exists()


def test_original_compute_contract_must_match_the_native_determinism_and_memory_settings(tmp_path, monkeypatch):
    value = original_state(tmp_path, monkeypatch)
    path = value['release_directory'] / 'preserved-checkpoints/latest.pt'
    state = torch.load(path, weights_only=True); state['compute_contract']['deterministic_training'] = False
    torch.save(state, path)
    with pytest.raises(ValueError, match='compute contract'): capture(value)


def test_a_completed_partial_state_capture_cannot_be_used_as_a_completed_training_parent(tmp_path, monkeypatch):
    value = original_state(tmp_path, monkeypatch); capture(value)
    with pytest.raises(ValueError):
        completed_parent(value['release_directory'] / 'preserved-checkpoints/adapter.pt', value['capture'] / 'manifest.json')


def test_source_checkpoint_bytes_changed_after_capture_are_rejected_before_target_creation(tmp_path, monkeypatch):
    value = original_state(tmp_path, monkeypatch); capture(value)
    path = value['release_directory'] / 'preserved-checkpoints/latest.pt'
    saved = torch.load(path, weights_only=True); saved['optimizer']['state'][0]['exp_avg'].add_(1.)
    torch.save(saved, path)
    with pytest.raises(ValueError):
        transition.prepare_transition(value['capture'] / 'manifest.json', value['config'], value['bundle'], retained_ranks=[0, 2])
    assert not value['bundle'].exists()


@pytest.mark.parametrize('ranks', [None, [0, 0], [0, 3], [False, 2]])
def test_original_rng_rank_mapping_must_be_explicit_distinct_and_in_range(tmp_path, monkeypatch, ranks):
    value = original_state(tmp_path, monkeypatch); capture(value)
    with pytest.raises(ValueError):
        transition.prepare_transition(value['capture'] / 'manifest.json', value['config'], value['bundle'], retained_ranks=ranks)
    assert not value['bundle'].exists()


def test_CPU_preparation_installs_a_resumable_state_without_starting_GPU_or_SGD(tmp_path, monkeypatch):
    arguments, value, calls = completed_resource_round(tmp_path, monkeypatch, micro=3)
    result = round_training.run_round(**arguments, prepare_only=True)
    assert result['status'] == 'prepared_auxiliary_round_without_GPU_or_SGD'
    assert result['resource_transition_source_step'] == 2 and calls == []
    assert not (value['target'] / 'manifest.json').exists()
    before = (value['target'] / 'resource-transition/latest.pt').read_bytes()
    completed = round_training.run_round(**arguments, resume=True)
    assert completed['training_resource_transition']['total_global_updates'] == 48
    assert calls == ['GPU_preflight', 'formal_SFT']
    assert (value['target'] / 'resource-transition/latest.pt').read_bytes() == before
