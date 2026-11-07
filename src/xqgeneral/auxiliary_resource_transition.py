"""Explicit full-state continuation from three auxiliary workers to two."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import torch

from . import auxiliary_policy_preflight as auxiliary
from .evidence import atomic_json, code_identity, digest, file_signature, manifest
from .finite_training import prepare_finite_epochs
from .full_decoder_topology import auxiliary_topology
from .recorded_search_inputs import BoundInputs
from .sft_preflight import full_parameter_summary, identical
from .training import compatible_resume, select_validation_rows, training_input_paths, training_memory_settings
from .training_index import IndexedTrainingRows
from .verified_inputs import input_session_from_environment

SOURCE_KIND = 'released_partial_auxiliary_training_state'
KIND = 'three_to_two_worker_auxiliary_state_transition'
PROFILE = 'first_pass_full_optimizer_three_to_two_workers_v1'
METADATA_CHANGES = {'auxiliary_policy_training.py', 'clean_sft_evaluation.py', 'collect_search.py',
    'distillation_preflight.py', 'distillation_training.py', 'distillation_validation.py',
    'full_decoder_topology.py', 'merge_search.py', 'search_distillation.py', 'search_pilot.py', 'sft_preflight.py'}
ADDED_MODULES = {'auxiliary_search_replay.py', 'oracle_policy_evaluation.py', 'search_parent.py',
                 'search_protection.py', 'auxiliary_resource_transition.py'}
RESOURCE_FIELDS = {'ddp_world_size', 'auxiliary_topology_profile', 'micro_batch_size', 'output', 'row_index_paths', 'purpose'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _weights(path, bound):
    bound.bind(path)
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def _bindings(proof, bound):
    _require(proof.get('status') == 'complete', 'Require completed bound evidence')
    for section in ['inputs', 'outputs']:
        for path, expected in proof[section].items():
            bound.bind(path, expected)


def _source_code(freeze_path, saved_code, bound):
    freeze = bound.json(freeze_path)
    source = Path(freeze['source_path']) / 'xqgeneral'
    original = freeze['source_sha256']
    _require(set(original) == {p.name for p in source.glob('*.py')}, 'Incomplete preserved execution source')
    for name, expected in original.items():
        _require(Path(name).name == name and name.endswith('.py'), 'Invalid preserved module name')
        _require(bound.bind(source / name)['sha256'] == expected, 'Preserved execution source changed')
    _require(saved_code.get('source_sha256') == {'src/xqgeneral/' + n: h for n, h in original.items()},
             'Checkpoint execution source differs from its complete preserved inventory')
    current = code_identity()
    target = {Path(n).name: h for n, h in current['source_sha256'].items()}
    changed = {n for n in original.keys() & target.keys() if original[n] != target[n]}
    _require(not original.keys() - target.keys() and changed <= METADATA_CHANGES and
             target.keys() - original.keys() <= ADDED_MODULES, 'Training computation changed across the resource transition')
    _require(all(n in original for n in ['training.py', 'training_parallel.py', 'finite_training.py',
             'training_index.py', 'modeling.py', 'model_inputs.py', 'sft.py']), 'Missing native training computation')
    return dict(original_code=saved_code, target_code=current, changed_metadata_modules=sorted(changed),
                added_modules=sorted(target.keys() - original.keys()), native_training_computation_byte_identical=True)


@contextmanager
def _data(config, bound):
    paths = training_input_paths(config)
    hashes = {str(p): bound.bind(p)['sha256'] for p in paths}
    roots = [Path(config['data_path']), *map(Path, config.get('replay_data_paths', []))]
    rows = {}
    try:
        for split in ['train', 'validation']:
            rows[split] = IndexedTrainingRows(config['row_index_paths'][split],
                [r / (split + '.jsonl') for r in roots], config['mixture'], split=split, verified_hashes=hashes)
        finite = prepare_finite_epochs(rows['train'], config['mixture'], config, config['ddp_world_size'])
        _require(finite is not None, 'Resource continuation requires the finite full-source traversal')
        yield rows, finite, hashes
    finally:
        for value in rows.values():
            value.close()


def _global_ids(rows, finite, step):
    pieces = [finite.batch_indices(step, rank) for rank in range(finite.world)]
    report = pieces[0][1]
    _require(all(note == report for _, note in pieces), 'Distributed batch reports differ')
    micro = finite.micro
    ordered = []
    for offset in range(0, len(pieces[0][0]), micro):
        for values, _ in pieces:
            ordered.extend(values[offset:offset + micro])
    return [rows.row_metadata(i)['id'] for i in ordered[:report['global_valid_rows']]], report


def _optimizer(saved):
    parameters = saved['trainable']; optimizer = saved['optimizer']; grouped = {}
    for name in parameters:
        group = 'board' if name.endswith('board_weight') else ('bridge' if name.startswith('bridges.') else 'decoder')
        grouped.setdefault(group, []).append(name)
    groups = optimizer['param_groups']
    _require([g.get('name') for g in groups] == list(grouped), 'Native Adam parameter groups changed')
    all_indices = []; steps = []
    for group in groups:
        names = grouped[group['name']]
        _require(len(names) == len(group['params']), 'Native Adam group coverage differs from full parameters')
        for name, index in zip(names, group['params'], strict=True):
            _require(type(index) is int and index in optimizer['state'], 'Missing full native Adam state')
            all_indices.append(index); item = optimizer['state'][index]
            _require(set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unsupported or incomplete native Adam state')
            step = item['step']
            _require(torch.is_tensor(step) and step.numel() == 1 and torch.isfinite(step).all() and
                     0 < float(step) <= saved['step'] and float(step).is_integer(), 'Invalid per-parameter Adam update count')
            steps.append(float(step))
            for field in ['exp_avg', 'exp_avg_sq']:
                tensor = item[field]
                _require(torch.is_tensor(tensor) and tensor.dtype == torch.float32 and
                         tensor.shape == parameters[name].shape and torch.isfinite(tensor).all(),
                         'Incompatible or nonfinite full Adam moments')
            _require((item['exp_avg_sq'] >= 0).all(), 'Negative Adam second moments')
    _require(len(all_indices) == len(set(all_indices)) == len(parameters) and
             set(all_indices) == set(optimizer['state']), 'Full native Adam index coverage changed')
    return dict(groups=len(groups), parameter_states=len(all_indices),
                minimum_parameter_step=min(steps), maximum_parameter_step=max(steps), all_moments_finite=True)


def _prefix(path, saved, finite):
    original = Path(path).read_bytes(); prefix = []; last = None
    for raw in original.splitlines(keepends=True):
        if len(prefix) == saved['step']:
            break
        row = json.loads(raw)
        _require(row.get('step') == len(prefix) + 1 and raw.endswith(b'\n'), 'Missing complete native training prefix')
        _require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ['loss', 'gradient_norm']),
                 'Nonfinite original native update log')
        _require(row.get('data_batch') == finite.batch_indices(row['step'] - 1, 0)[1], 'Logged finite data cursor changed')
        prefix.append(raw); last = row
    _require(len(prefix) == saved['step'] and last['supervised_tokens'] == saved['tokens_seen'],
             'Native checkpoint tokens or contiguous update prefix differ')
    return b''.join(prefix)


def _snapshot(round_directory, release_directory, source_freeze, bound):
    root, release = Path(round_directory), Path(release_directory)
    contract = bound.json(root / 'contract.json'); recipe = bound.json(root / 'recipe.json')
    _require(not (root / 'manifest.json').exists(), 'Capture an interrupted auxiliary round, not a completed model')
    end = bound.json(release / 'release-verification.json'); plan = bound.json(release / 'release-plan.json')
    preserved = bound.json(release / 'preserved-checkpoints.json')
    _require(end.get('project_GPU_workers_all_terminal') is True and
             end.get('parent_attempt_interrupted_for_user_resource_request') is True and
             end.get('other_existing_GPU_occupants_not_signalled') is True,
             'Require the completed explicit user resource release')
    _require(Path(plan['training_output']).resolve() == (root / 'sft').resolve(), 'Release belongs to another round')
    _require(plan.get('world_size') == 3 and len(plan['GPU_workers']) == 3 and
             sorted(p['rank_environment']['RANK'] for p in plan['GPU_workers']) == ['0', '1', '2'],
             'Release must identify all three original worker ranks')
    for item in [plan['launcher'], *plan['GPU_workers']]:
        proc = Path('/proc') / str(item['pid'])
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            same = int(fields[19]) == item['start_time'] and hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest() == item['command_sha256']
        except FileNotFoundError:
            same = False
        _require(not same or fields[0] in ['Z', 'X'], 'Original GPU workers have not actually stopped')
    latest_path = release / 'preserved-checkpoints/latest.pt'
    adapter_path = release / 'preserved-checkpoints/adapter.pt'
    _require(preserved['files']['latest.pt']['path'] == str(latest_path) and
             preserved['files']['adapter.pt']['path'] == str(adapter_path), 'Preserved checkpoint paths changed')
    saved, selected = _weights(latest_path, bound), _weights(adapter_path, bound)
    required = {'config', 'world_size', 'step', 'trainable', 'optimizer', 'best_step', 'best_loss',
                'initial_loss', 'patience_count', 'tokens_seen', 'input_hashes', 'code',
                'compute_contract', 'data_state', 'random_state', 'torch_rng_state', 'cuda_rng_states'}
    _require(isinstance(saved, dict) and required <= saved.keys() and
             isinstance(selected, dict) and {'config', 'selected_step', 'code', 'input_hashes', 'trainable'} <= selected.keys(),
             'Incomplete native checkpoint state')
    config = bound.json(root / 'sft/config.json')
    _require(saved['config'] == selected['config'] == config and saved['world_size'] == 3 and
             auxiliary_topology(config)['world_size'] == 3, 'Require the original three-worker auxiliary state')
    step = saved['step']
    _require(type(step) is int and 0 < step < config['steps'] and
             step == end['durable_native_checkpoint_step'] == preserved['saved_step'], 'Invalid saved partial update count')
    _require(type(saved['best_step']) is int and 0 <= saved['best_step'] <= step and
             type(saved['patience_count']) is int and saved['patience_count'] >= 0 and
             type(saved['tokens_seen']) is int and saved['tokens_seen'] > 0,
             'Invalid original selection, patience or token counter')
    _require(saved['code'] == selected['code'] == contract['code'], 'Saved checkpoint execution identity changed')
    deterministic = config.get('deterministic_training', False)
    compute = saved['compute_contract']; memory = training_memory_settings(config)
    _require(isinstance(compute, dict) and compute.get('deterministic_training') is deterministic and
             compute.keys() == {'deterministic_training', 'cublas_workspace_config', *memory} and
             all(compute[k] == v for k, v in memory.items()) and
             (compute.get('cublas_workspace_config') in {':4096:8', ':16:8'} if deterministic else
              compute.get('cublas_workspace_config') is None), 'Original deterministic compute contract changed')
    code = _source_code(source_freeze, saved['code'], bound)
    token = bound.json(contract['token_preflight']); template = bound.json(contract['recipe_template'])
    _bindings(token, bound)
    compiled, expected, budget = auxiliary.compile_recipe(token, template, root)
    _require(compiled == recipe == contract['compiled_recipe'] and expected == config and
             digest(contract['source']) == contract['source_sha256'] and
             digest(contract['recipe_template']) == contract['recipe_template_sha256'] and
             digest(contract['token_preflight']) == contract['token_preflight_sha256'], 'Original full-data round contract changed')
    reference = _weights(contract['source'], bound)['trainable']
    _require(list(saved['trainable']) == list(reference) == list(selected['trainable']), 'Original full parameter ordering changed')
    summary = full_parameter_summary(saved['trainable'], reference)
    full_parameter_summary(selected['trainable'], reference)
    optimizer = _optimizer(saved)
    _require(selected['selected_step'] == saved['best_step'] and
             saved['input_hashes'] == selected['input_hashes'], 'Original selected best state differs from resume state')
    _require(all(type(saved[k]) in (int, float) and math.isfinite(saved[k]) for k in ['initial_loss', 'best_loss']),
             'Invalid original validation loss state')
    try:
        random.Random().setstate(saved['random_state'])
        torch.Generator(device='cpu').set_state(saved['torch_rng_state'])
    except (ValueError, RuntimeError, TypeError) as error:
        raise ValueError('Invalid original native random state') from error
    _require(len(saved['cuda_rng_states']) == 3 and all(torch.is_tensor(s) and s.dtype == torch.uint8 and
             s.ndim == 1 and s.numel() > 0 for s in saved['cuda_rng_states']), 'Missing original worker RNG states')
    with _data(config, bound) as (rows, finite, hashes):
        _require(saved['input_hashes'] == hashes, 'Original native training input hashes changed')
        finite.check_resume(saved['data_state'], step)
        _require(saved['data_state']['epoch'] == 0 and saved['data_state']['padding_rows_consumed'] == 0,
                 'This transition profile requires the first pass before its padded tail')
        ids = bound.json(root / 'sft/validation_ids.json')
        planned = [r['id'] for r in select_validation_rows(rows['validation'], config['validation_stages'],
                   config['seed'], config['validation_examples'])]
        _require(ids == planned, 'Original full validation IDs or order changed')
        prefix_path = root / 'sft/training.jsonl'; bound.bind(prefix_path)
        prefix = _prefix(prefix_path, saved, finite)
    bound.unchanged()
    verification = dict(profile=PROFILE, saved_step=step, selected_step=saved['best_step'],
        source_world_size=3, completed_auxiliary_round=False, **summary, optimizer=optimizer,
        full_native_input_hashes_and_first_pass_cursor_checked=True,
        contiguous_original_update_log_rows=step, original_training_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        original_full_validation_ids_and_order_preserved=True, training_budget=budget,
        source_code_transition=code, new_model_or_engine_queries=0, independent_test_used=False)
    return dict(saved=saved, selected=selected, config=config, prefix=prefix, validation_ids=ids,
                verification=verification, contract=contract)


def capture_source(round_directory, release_directory, source_freeze, output, *, resume=False):
    config = dict(round_directory=str(Path(round_directory).resolve()), release_directory=str(Path(release_directory).resolve()),
                  source_freeze=str(Path(source_freeze).resolve()), profile=PROFILE)
    root = Path(output).resolve(); bound = BoundInputs()
    if root.exists():
        _require(resume, 'Preserve an existing source-state capture')
        proof = bound.json(root / 'manifest.json')
        _require(proof.get('kind') == SOURCE_KIND and proof.get('config') == config, 'Source capture contract changed')
        _bindings(proof, bound)
    else:
        proof = None
    observed = _snapshot(config['round_directory'], config['release_directory'], config['source_freeze'], bound)
    if proof is not None:
        _require(proof['verification'] == observed['verification'], 'Captured partial state no longer reconstructs')
        return proof['verification']
    root.mkdir(parents=True)
    result = manifest(SOURCE_KIND, config, outputs=[], verification=observed['verification'])
    result['inputs'] = bound.artifacts
    atomic_json(root / 'manifest.json', result)
    return result['verification']


def _captured(path, bound):
    proof = bound.json(path)
    _require(proof.get('kind') == SOURCE_KIND and proof.get('config', {}).get('profile') == PROFILE,
             'Require the explicit released partial auxiliary state')
    _bindings(proof, bound)
    c = proof['config']; observed = _snapshot(c['round_directory'], c['release_directory'], c['source_freeze'], bound)
    _require(observed['verification'] == proof['verification'], 'Released partial state certificate differs from native reconstruction')
    return proof, observed


def source_spec(source_manifest, parent, token_manifest, retained_ranks, bound):
    """Bind a released source separately from mutable native checkpoint links."""
    source_manifest = Path(source_manifest).resolve()
    _require(isinstance(retained_ranks, (list, tuple)) and len(retained_ranks) == 2 and
             all(type(r) is int and 0 <= r < 3 for r in retained_ranks) and len(set(retained_ranks)) == 2,
             'Declare the two retained original RNG ranks')
    proof, observed = _captured(source_manifest, BoundInputs())
    contract = observed['contract']
    _require(contract['source'] == str(Path(parent).resolve()) and
             contract['token_preflight'] == str(Path(token_manifest).resolve()),
             'Released state belongs to another completed parent or token corpus')
    binding = bound.bind(source_manifest)
    return dict(path=str(source_manifest), sha256=binding['sha256'], saved_step=proof['verification']['saved_step'],
                retained_ranks=list(retained_ranks), profile=PROFILE)


def _transition(source_manifest, target_config, retained_ranks, bound):
    source, captured = _captured(source_manifest, bound)
    original, target = captured['config'], target_config
    _require(auxiliary_topology(target)['world_size'] == 2, 'Target must declare the explicit two-worker auxiliary profile')
    _require(set(original) == set(target) and all(original[k] == target[k] for k in original.keys() - RESOURCE_FIELDS),
             'Resource continuation changed model, data, optimizer or update schedule')
    normalized = dict(original, **{k: target[k] for k in RESOURCE_FIELDS if k in target})
    compatible_resume(normalized, target)
    _require(isinstance(retained_ranks, (list, tuple)) and len(retained_ranks) == 2 and
             all(type(r) is int and 0 <= r < 3 for r in retained_ranks) and len(set(retained_ranks)) == 2,
             'Declare the two retained original RNG ranks in their new order')
    old = captured['saved']; step = old['step']
    with _data(original, bound) as (old_rows, old_finite, _), _data(target, bound) as (rows, finite, hashes):
        old_contract, new_contract = old_finite.contract(), finite.contract()
        changing = {'world_size', 'micro_batch_size', 'padded_rows_per_epoch'}
        _require(old_contract.keys() == new_contract.keys() and
                 all(old_contract[k] == new_contract[k] for k in old_contract.keys() - changing),
                 'Finite source selection changed across worker counts')
        for ordinal in range(step + 1):
            before, _ = _global_ids(old_rows['train'], old_finite, ordinal)
            after, _ = _global_ids(rows['train'], finite, ordinal)
            _require(before == after, 'Consumed or next global optimizer batch changed')
        selected_ids = [r['id'] for r in select_validation_rows(rows['validation'], target['validation_stages'],
                       target['seed'], target['validation_examples'])]
        _require(selected_ids == captured['validation_ids'], 'Target validation IDs or order changed')
        state = dict(old, config=deepcopy(target), world_size=2, code=code_identity(), input_hashes=hashes,
                     data_state=finite.state_at(step), cuda_rng_states=[old['cuda_rng_states'][r] for r in retained_ranks])
    best = dict(captured['selected'], config=deepcopy(target), code=state['code'], input_hashes=hashes)
    evidence = dict(profile=PROFILE, source_saved_step=step, target_world_size=2, source_world_size=3,
        source_micro_batch_size=original['micro_batch_size'], target_micro_batch_size=target['micro_batch_size'],
        retained_original_RNG_ranks=list(retained_ranks), all_parameters_and_Adam_values_preserved=True,
        per_parameter_update_counts_and_global_schedule_preserved=True,
        consumed_global_batches_and_next_valid_row_order_compared=step + 1,
        original_full_validation_ids_and_order_preserved=True, first_pass_unique_rows_consumed=state['data_state']['unique_rows_consumed'],
        target_future_padding_rows_per_pass=state['data_state']['contract']['padded_rows_per_epoch'],
        source_future_padding_rows_per_pass=old['data_state']['contract']['padded_rows_per_epoch'],
        exact_future_three_worker_trajectory_claimed=False, native_same_world_resume_guards_changed=False,
        new_SGD_updates=0, completed_auxiliary_round=False, source_manifest_sha256=digest(source_manifest),
        source_code_transition=captured['verification']['source_code_transition'])
    return state, best, captured['prefix'], selected_ids, evidence


def _checked_bundle(directory, source_manifest, target_config, retained_ranks, bound):
    proof = bound.json(Path(directory) / 'manifest.json')
    _require(proof.get('kind') == KIND and proof.get('config') == dict(profile=PROFILE,
        source_manifest=str(Path(source_manifest).resolve()), target_config=target_config,
        retained_ranks=list(retained_ranks)), 'Resource transition bundle contract changed')
    _bindings(proof, bound)
    state, best, prefix, ids, verification = _transition(source_manifest, target_config, retained_ranks, bound)
    root = Path(directory)
    _require(identical(_weights(root / 'latest.pt', bound), state) and
             identical(_weights(root / 'adapter.pt', bound), best), 'Full parameter or Adam transition values changed')
    _require((root / 'training-prefix.jsonl').read_bytes() == prefix and
             bound.json(root / 'validation_ids.json') == ids and proof['verification'] == verification,
             'Resource transition prefix, validation IDs or verification changed')
    bound.unchanged()
    return verification


def prepare_transition(source_manifest, target_config, output, *, retained_ranks, resume=False, bound=None):
    root = Path(output).resolve(); source_manifest = Path(source_manifest).resolve()
    bound = bound or BoundInputs()
    if root.exists():
        _require(resume, 'Preserve an existing resource transition attempt')
        return _checked_bundle(root, source_manifest, target_config, retained_ranks, bound)
    state, best, prefix, ids, result = _transition(source_manifest, target_config, retained_ranks, bound)
    root.mkdir(parents=True)
    torch.save(state, root / 'latest.pt'); torch.save(best, root / 'adapter.pt')
    (root / 'training-prefix.jsonl').write_bytes(prefix); atomic_json(root / 'validation_ids.json', ids)
    _require(identical(_weights(root / 'latest.pt', bound), state) and identical(_weights(root / 'adapter.pt', bound), best),
             'Written full optimizer transition failed native tensor readback')
    bound.unchanged()
    proof = manifest(KIND, dict(profile=PROFILE, source_manifest=str(source_manifest), target_config=target_config,
        retained_ranks=list(retained_ranks)), outputs=[root / n for n in ['latest.pt', 'adapter.pt', 'training-prefix.jsonl', 'validation_ids.json']], verification=result)
    # Outputs must remain separately mutable only through a native atomic replace.
    proof['inputs'] = {p: v for p, v in bound.artifacts.items() if p not in proof['outputs']}
    atomic_json(root / 'manifest.json', proof)
    return result


def checked_transition(directory, bound=None):
    bound = bound or BoundInputs(); proof = bound.json(Path(directory) / 'manifest.json'); c = proof['config']
    return _checked_bundle(directory, c['source_manifest'], c['target_config'], c['retained_ranks'], bound)


def _link_checkpoint(source, target, bound):
    expected = bound.bind(source); before = file_signature(source)
    os.link(source, target)
    after = file_signature(source)
    _require(before[:4] == after[:4] and source.samefile(target), 'Checkpoint identity changed while linking')
    _require(digest(source) == expected['sha256'] and file_signature(source) == after,
             'Checkpoint bytes changed while installing its native link')
    # The owned link operation changes inode ctime without changing any bytes.
    # This phase's guard may acknowledge only that verified transition. Later
    # native atomic replacements are checked by a new readback, not this guard.
    bound.signatures[str(source)] = after


def install_transition(directory, output):
    root, target = Path(directory).resolve(), Path(output).resolve(); bound = BoundInputs()
    result = checked_transition(root, bound); proof = bound.json(root / 'manifest.json')
    _require(Path(proof['config']['target_config']['output']).resolve() == target, 'Native output path differs from transition')
    marker = dict(transition_manifest=str(root / 'manifest.json'), transition_manifest_sha256=digest(root / 'manifest.json'),
                  initial_resume_step=result['source_saved_step'])
    if target.exists():
        _require(bound.json(target / 'resource-transition-installed.json') == marker,
                 'Existing native output has no matching transition installation')
        return result
    target.mkdir(parents=True)
    for name in ['latest.pt', 'adapter.pt']:
        _link_checkpoint(root / name, target / name, bound)
    shutil.copyfile(root / 'training-prefix.jsonl', target / 'training.jsonl')
    shutil.copyfile(root / 'validation_ids.json', target / 'validation_ids.json')
    atomic_json(target / 'config.json', proof['config']['target_config'])
    atomic_json(target / 'resource-transition-installed.json', marker)
    bound.unchanged()
    return result


def transition_ledger(directory, reference, native_directory, config, metrics, producer_code, bound):
    """Reconcile a completed CPU transition with its actual native continuation."""
    root, native = Path(directory).resolve(), Path(native_directory).resolve()
    proof = bound.json(root / 'manifest.json'); c = proof.get('config', {}); result = proof.get('verification', {})
    source_path = Path(reference['path']); source = bound.json(source_path)
    _require(bound.artifacts[str(source_path)]['sha256'] == reference['sha256'] and
             source.get('status') == 'complete' and source.get('kind') == SOURCE_KIND and
             source.get('config', {}).get('profile') == PROFILE, 'Completed transition lost its released source binding')
    _require(proof.get('status') == 'complete' and proof.get('kind') == KIND and proof.get('code') == producer_code and
             c == dict(profile=PROFILE, source_manifest=str(source_path), target_config=config,
                       retained_ranks=reference['retained_ranks']), 'Completed transition lost its native target contract')
    step = reference['saved_step']
    marker = bound.json(native / 'resource-transition-installed.json')
    _require(marker == dict(transition_manifest=str(root / 'manifest.json'),
             transition_manifest_sha256=digest(root / 'manifest.json'), initial_resume_step=step),
             'Native continuation lost its original transition installation')
    _require(source['verification']['saved_step'] == result.get('source_saved_step') == step and
             result.get('all_parameters_and_Adam_values_preserved') is True and
             result.get('per_parameter_update_counts_and_global_schedule_preserved') is True and
             result.get('original_full_validation_ids_and_order_preserved') is True and
             result.get('consumed_global_batches_and_next_valid_row_order_compared') == step + 1 and
             result.get('new_SGD_updates') == 0 and result.get('exact_future_three_worker_trajectory_claimed') is False,
             'Completed transition lacks its qualified full-state preservation checks')
    _require(source['verification']['source_code_transition']['target_code'] == producer_code,
             'Released source capture and target execution source differ')
    execution = bound.json(native / 'execution.json')
    inputs = {str(p): bound.bind(p)['sha256'] for p in training_input_paths(config)}
    _require(execution.get('resumed_from_step') == step and execution.get('code') == producer_code and
             execution.get('input_hashes') == inputs and metrics['steps_completed'] > step and
             metrics['world_size'] == config['ddp_world_size'] == 2, 'Actual native training did not continue the preserved global step')
    return dict(profile=PROFILE, source_manifest=str(source_path), source_manifest_sha256=reference['sha256'],
        transition_manifest=str(root / 'manifest.json'), transition_manifest_sha256=digest(root / 'manifest.json'),
        original_three_worker_updates=step, new_two_worker_updates=metrics['steps_completed'] - step,
        total_global_updates=metrics['steps_completed'], retained_original_RNG_ranks=reference['retained_ranks'],
        source_micro_batch_size=result['source_micro_batch_size'], target_micro_batch_size=result['target_micro_batch_size'],
        full_state_CPU_comparison_receipt_reused=True, different_world_future_bitwise_equivalence_claimed=False)


def completed_transition(proof, native_directory, config, metrics, bound):
    """Read a historical transition ledger without re-executing artifact code."""
    reference = proof.get('config', {}).get('resource_source')
    recorded = proof.get('verification', {}).get('training_resource_transition')
    if reference is None:
        _require(recorded is None, 'Resource transition ledger has no source contract')
        return None
    _require(isinstance(recorded, dict), 'Completed resource continuation needs its actual ledger')
    actual = transition_ledger(Path(recorded['transition_manifest']).parent, reference, native_directory,
                               config, metrics, proof['config']['code'], bound)
    _require(actual == recorded, 'Completed resource continuation differs from actual native execution')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    capture = sub.add_parser('capture')
    for name in ['round', 'release', 'source-freeze', 'output']:
        capture.add_argument('--' + name, required=True)
    capture.add_argument('--resume', action='store_true'); args = parser.parse_args()
    with input_session_from_environment():
        print(json.dumps(capture_source(args.round, args.release, args.source_freeze, args.output, resume=args.resume)), flush=True)


if __name__ == '__main__':
    main()
