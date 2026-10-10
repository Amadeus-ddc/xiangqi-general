"""Full-decoder four-worker update, memory and exact-continuation preflight."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import torch

from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest, write_jsonl
from .foundation_handoff import Artifacts
from .modeling import expanded_initialization_reference
from .sft import checked_parent, prepare_config, training_command
from .training import atomic_checkpoint, compatible_resume, training_input_paths
from .feature_store import feature_paths
from .training_index import build_index, index_paths, IndexedTrainingRows

RESUME_FIELDS = ['step', 'world_size', 'trainable', 'optimizer', 'random_state', 'torch_rng_state',
    'cuda_rng_states', 'tokens_seen', 'initial_loss', 'input_hashes', 'code', 'compute_contract',
    'patience_count', 'best_step', 'best_loss']


def identical(a, b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(identical(v, w) for v, w in zip(a, b))
    return type(a) is type(b) and a == b


def compare_resume(continuous, resumed):
    roots = [Path(continuous), Path(resumed)]
    a, b = [torch.load(p / 'latest.pt', map_location='cpu', weights_only=True, mmap=True) for p in roots]
    fields = RESUME_FIELDS + (['data_state'] if 'data_state' in a or 'data_state' in b else [])
    failures = [k for k in fields if k not in a or k not in b or not identical(a[k], b[k])]
    logs = [[{k: v for k, v in json.loads(line).items() if k != 'seconds'}
             for line in (p / 'training.jsonl').read_text().splitlines()] for p in roots]
    if logs[0] != logs[1]:failures.append('step_logs')
    differences = []
    for name in set(a['trainable']) | set(b['trainable']):
        left, right = a['trainable'].get(name), b['trainable'].get(name)
        if identical(left, right):continue
        differences.append(float((left - right).abs().max()) if torch.is_tensor(left) and
                           torch.is_tensor(right) and left.shape == right.shape else float('inf'))
    return {'status': 'failed' if failures else 'complete', 'mismatched_fields': failures,
            'mismatched_trainable_tensors': len(differences), 'max_parameter_absolute_error': max(differences, default=0),
            'steps': [a['step'], b['step']], 'world_sizes': [a['world_size'], b['world_size']]}


def checked_artifacts(proof):
    if proof.get('status') != 'complete':raise ValueError('Completed evidence is required')
    for path, item in {**proof['inputs'], **proof['outputs']}.items():
        if Path(path).stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Completed preflight evidence changed: ' + path)


def full_parameter_summary(state, initial):
    if set(state) != set(initial):raise ValueError('Full-decoder parameter set differs from its initial reference')
    for name, tensor in state.items():
        if tensor.dtype != torch.float32 or tensor.shape != initial[name].shape or not torch.isfinite(tensor).all():
            raise ValueError('Full-decoder state has incompatible or nonfinite FP32 parameters')
    return {'trainable_tensors': len(state), 'trainable_parameters': sum(t.numel() for t in state.values()),
            'complete_shapes_and_finite_fp32_verified': True}


def probe_data(recipe, token_proof, output):
    lengths = load_jsonl(next(p for p in token_proof['outputs'] if p.endswith('token-lengths.jsonl')))
    result = {}
    for split, limit in [('train', 24), ('validation', 12)]:
        source = load_jsonl(Path(recipe['data_path']) / (split + '.jsonl'))
        by_id = {row['id']: row for row in source}
        selected = sorted([item for item in lengths if item['split'] == split],
                          key=lambda item: (-item['tokens'], item['id']))[:limit]
        if len(selected) != limit or len({item['id'] for item in selected}) != limit:
            raise ValueError('The actual longest-sequence probe needs 24 train and 12 validation labels')
        rows = [by_id[item['id']] for item in selected]
        if any(row['split'] != split or row['stage'] != 'explanation' for row in rows):
            raise ValueError('Probe labels must preserve their actual protected split and stage')
        write_jsonl(Path(output) / (split + '.jsonl'), rows)
        result[split] = {'ids': [row['id'] for row in rows], 'maximum_tokens': max(item['tokens'] for item in selected)}
    return result


def probe_row_indexes(common, root):
    """Bind cloned longest-example sources instead of reusing full-corpus offsets."""
    if 'row_index_paths' not in common:
        return common
    roots = [Path(common['data_path']), *[Path(p) for p in common.get('replay_data_paths', [])]]
    indexes = {}
    for split in common['row_index_paths']:
        directory = Path(root) / (split + '-probe-index')
        paths = [p / (split + '.jsonl') for p in roots]
        if not directory.exists():
            build_index(paths, directory, split=split)
        else:
            IndexedTrainingRows(directory, paths, common['mixture'], split=split).close()
        indexes[split] = str(directory)
    return dict(common, row_index_paths=indexes)


def verify_updated_probe(root, initial_path, expected_inputs, config, execution_code):
    root = Path(root)
    comparison = compare_resume(root / 'continuous', root / 'resumed')
    if comparison['status'] != 'complete' or comparison['steps'] != [4, 4] or comparison['world_sizes'] != [4, 4]:
        raise ValueError('Actual four-worker full-decoder continuation differs')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)['trainable']
    actual = torch.load(root / 'continuous/latest.pt', map_location='cpu', weights_only=True, mmap=True)
    compatible_resume(actual['config'], config)
    if actual['code'] != execution_code or actual['input_hashes'] != expected_inputs or len(actual['cuda_rng_states']) != 4:
        raise ValueError('Full-decoder probe execution or immutable inputs differ')
    full_parameter_summary(actual['trainable'], initial)
    changes = dict.fromkeys(['decoder', 'bridge', 'board'], 0)
    for name, tensor in actual['trainable'].items():
        if tensor.dtype != torch.float32 or tensor.shape != initial[name].shape or not torch.isfinite(tensor).all():
            raise ValueError('Full-decoder probe has invalid FP32 parameters')
        group = 'board' if name.endswith('board_weight') else ('bridge' if name.startswith('bridges.') else 'decoder')
        changes[group] += int(not torch.equal(tensor, initial[name]))
    if not all(changes.values()):raise ValueError('Decoder, bridge and board parameters must all actually update')
    states = actual['optimizer']['state']
    if len(states) != len(initial) or any(float(s['step']) != 4 or
            any(not torch.isfinite(s[k]).all() or s[k].dtype != torch.float32 for k in ['exp_avg', 'exp_avg_sq'])
            for s in states.values()):
        raise ValueError('Full-decoder Adam states must be complete, finite FP32 at actual step four')
    grouped = {group: [name for name in initial if ('board' if name.endswith('board_weight') else
               ('bridge' if name.startswith('bridges.') else 'decoder')) == group] for group in changes}
    seen = set()
    for group in actual['optimizer']['param_groups']:
        names = grouped.get(group['name'], [])
        if len(group['params']) != len(names):raise ValueError('Adam parameter grouping differs from the actual model')
        for index, name in zip(group['params'], names):
            if index in seen or states[index]['exp_avg'].shape != initial[name].shape or states[index]['exp_avg_sq'].shape != initial[name].shape:
                raise ValueError('Adam state shapes differ from the actual full-decoder parameters')
            seen.add(index)
    if seen != set(states):raise ValueError('Adam states are not covered exactly once by actual parameter groups')
    rows = load_jsonl(root / 'continuous/training.jsonl')
    if [r['step'] for r in rows] != [1, 2, 3, 4] or any(not math.isfinite(r['loss']) or
            not math.isfinite(r['gradient_norm']) or r['gradient_norm'] <= 0 for r in rows):
        raise ValueError('Every probe update needs finite loss and nonzero finite gradients')
    resources = []
    for run, step in [('continuous', 4), ('resumed', 2), ('resumed', 4)]:
        for rank in range(4):
            item = json.loads((root / run / f'resources-step-{step}-rank-{rank}.json').read_text())
            if (item['rank'] != rank or item['world_size'] != 4 or item['actual_step'] != step or
                    item.get('training_main_returned_successfully') is not True or
                    not 0 < item['peak_allocated_gib'] <= item['total_memory_gib'] or
                    not math.isfinite(item['total_memory_gib'])):
                raise ValueError('Each actual GPU worker must report a successful finite memory measurement')
            resources.append(dict(item, run=run))
    return {'exact_resume': comparison, 'updated_tensors_by_group': changes,
            'trainable_tensors': len(initial), 'trainable_parameters': sum(t.numel() for t in initial.values()),
            'all_trainable_and_adam_states_finite_fp32': True, 'all_four_update_gradients_finite_nonzero': True,
            'actual_gpu_worker_resources': resources}


def run_preflight(source, recipe_path, token_manifest, output, *, resume=False):
    root = Path(output);recipe_path = Path(recipe_path);token_manifest = Path(token_manifest)
    recipe = json.loads(recipe_path.read_text());saved = checked_parent(source, True)
    formal = prepare_config(saved, recipe, source, root / 'unused-formal-output')
    if (formal.get('decoder_training') != 'full' or formal.get('trainable_parameter_dtype') != 'float32' or
            formal.get('ddp_world_size') != 4 or formal['batch_size'] != 16 or formal.get('micro_batch_size') != 1 or
            not formal.get('deterministic_training') or not formal.get('feature_cache_mmap')):
        raise ValueError('Preflight requires the exact deterministic four-worker full-decoder recipe')
    token = json.loads(token_manifest.read_text());checked_artifacts(token)
    if token.get('kind') != 'latent_only_reviewed_explanation_data_preflight' or token['config']['data'] != recipe['data_path']:
        raise ValueError('A completed token preflight for the exact reviewed data is required')
    if (token['verification']['token_limit'] != formal['max_tokens'] or
            not token['verification']['all_train_validation_supervision_fits_without_truncation']):
        raise ValueError('Formal token limit differs from the completed no-truncation preflight')
    cache_item = token['cache_header_only_input']
    if formal['feature_path'] != cache_item['path'] or formal['feature_path'] != saved['feature_path']:
        raise ValueError('Full-decoder preflight must use the exact foundation feature cache')
    foundation_path = Path(token['config']['foundation_preflight'])
    foundation = json.loads(foundation_path.read_text())
    if foundation.get('status') != 'complete':raise ValueError('Completed pinned-base foundation preflight is required')
    base_assets = {p: item for p, item in foundation['inputs'].items()
                   if Path(p).resolve().parent == Path(formal['model_path']).resolve()}
    if not any(p.endswith('.safetensors') for p in base_assets):raise ValueError('Pinned base weights are not hash-bound')
    checked_artifacts({'status': 'complete', 'inputs': base_assets, 'outputs': {}})
    contract = {'recipe_sha256': digest(recipe_path), 'source_sha256': digest(source),
        'source_config_sha256': digest(Path(source).parent / 'config.json'),
        'token_preflight_sha256': digest(token_manifest), 'base_assets': base_assets, 'code': code_identity()}
    cache = Artifacts()
    cache.add(formal['feature_path'], {key: cache_item[key] for key in ['sha256', 'bytes']})
    for path in feature_paths(formal['feature_path'])[1:]:
        name = str(path)
        if name not in token['inputs']:
            raise ValueError('Token preflight must bind every feature storage shard')
        cache.add(name, token['inputs'][name])
    cache.verify()
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Preserve an existing preflight; resume only its exact immutable contract')
        if (root / 'manifest.json').exists():
            proof = json.loads((root / 'manifest.json').read_text());checked_artifacts(proof);return proof['verification']
    else:root.mkdir(parents=True);atomic_json(root / 'contract.json', contract)
    selected = probe_data(recipe, token, root / 'data')
    common = dict(formal, data_path=str(root / 'data'), steps=8, min_steps=8, eval_every=2, patience=8,
                  validation_examples=12, generation_examples=0, purpose='Four-GPU full-decoder actual 4 versus 2+2 probe; no independent test or completed SFT claim')
    common = probe_row_indexes(common, root)
    expected_inputs = {str(p): cache.expected[str(p)]['sha256'] if str(p) in cache.expected else digest(p)
                       for p in training_input_paths(common)}
    initial = root / 'zero-init.pt';initial_manifest = root / 'zero-init.manifest.json'
    if not initial.exists():
        torch.manual_seed(common['seed'])
        state = expanded_initialization_reference(common, source)
        atomic_checkpoint(initial, {'trainable': state, 'config': common, 'code': contract['code']})
        del state
        atomic_json(initial_manifest, manifest('expanded_full_decoder_initial_reference', common,
            [Path(source), Path(source).parent / 'config.json', *[Path(p) for p in base_assets]], [initial],
            {'all_foundation_values_preserved': True, 'gpu_training_started': False}))
    initial_proof = json.loads(initial_manifest.read_text());checked_artifacts(initial_proof)
    if initial_proof['config'] != common or initial_proof['code'] != contract['code']:
        raise ValueError('Preserved CPU initialization reference differs from the exact probe')
    cache.unchanged()
    for run, target in [('continuous', 4), ('resumed', 2), ('resumed', 4)]:
        run_root = root / run;config = dict(common, output=str(run_root));path = root / (run + '.config.json')
        if path.exists() and json.loads(path.read_text()) != config:raise ValueError('Changed actual probe configuration')
        atomic_json(path, config)
        latest = run_root / 'latest.pt';step = 0
        if latest.exists():
            previous = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
            compatible_resume(previous['config'], config)
            if previous['code'] != contract['code'] or previous['input_hashes'] != expected_inputs or previous['world_size'] != 4:
                raise ValueError('Probe continuation inputs or executed source changed')
            step = previous['step'];del previous
        if step < target:
            command = training_command(config, path, latest if step else None)
            command[command.index('xqgeneral.train')] = 'xqgeneral.sft_probe_worker'
            command += ['--stop-after', str(target)]
            with (root / (run + '.log')).open('a') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        cache.unchanged()
    observed = verify_updated_probe(root, initial, expected_inputs, dict(common, output=str(root / 'continuous')), contract['code'])
    checked_artifacts(initial_proof);checked_artifacts(token)
    checked_artifacts({'status': 'complete', 'inputs': base_assets, 'outputs': {}})
    cache.unchanged()
    if digest(recipe_path) != contract['recipe_sha256'] or digest(source) != contract['source_sha256']:
        raise ValueError('Formal recipe or selected foundation changed during actual GPU preflight')
    result = {'status': 'complete', 'evidence_state': 'reconstructed_baseline', **observed,
        'full_decoder_initialization_preserved_all_foundation_tensors': True,
        'actual_continuous_four_and_two_plus_two_updates_compared': True,
        'longest_actual_train_validation_probe': selected,
        'independent_test_used_for_training_or_selection': False,
        'formal_sft_started_by_this_preflight': False, 'strong_play_or_coaching_gain_proven': False}
    atomic_json(root / 'verification.json', result)
    outputs = [initial, initial_manifest, root / 'verification.json', root / 'data/train.jsonl', root / 'data/validation.jsonl',
               root / 'continuous.config.json', root / 'resumed.config.json']
    outputs += [p for directory in common.get('row_index_paths', {}).values() for p in index_paths(directory)]
    for run in ['continuous', 'resumed']:
        outputs += [root / run / name for name in ['latest.pt', 'training.jsonl', 'config.json', 'execution.json']]
        outputs += sorted((root / run).glob('resources-step-*-rank-*.json'))
    inputs = [recipe_path, token_manifest, foundation_path, Path(source), Path(source).parent / 'manifest.json',
              Path(source).parent / 'config.json', *[Path(p) for p in token['inputs']], *[Path(p) for p in token['outputs']],
              *[Path(p) for p in base_assets]]
    proof = manifest('four_gpu_clean_full_decoder_sft_preflight',
        {'recipe': str(recipe_path), 'source': str(source), 'token_preflight': str(token_manifest)},
        list(dict.fromkeys(inputs)), outputs, result)
    proof['fresh_feature_cache_input'] = dict(cache_item, fresh_full_bytes_hash_verified_by_this_preflight=True)
    atomic_json(root / 'manifest.json', proof)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', required=True);parser.add_argument('--recipe', required=True)
    parser.add_argument('--token-preflight', required=True);parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true');args = parser.parse_args()
    torch.set_num_threads(8)
    print(json.dumps(run_preflight(args.init, args.recipe, args.token_preflight, args.output, resume=args.resume)), flush=True)


if __name__ == '__main__':main()
