"""Add immutable recorded-history expert features without rewriting course data."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .evidence import atomic_json, digest, history_key, manifest, write_jsonl
from .expert import FrozenPx0, encode_history
from .rules import replay
from .symmetry import mirror_fen, mirror_move


def validate_cache(cache, depths):
    keys = cache['keys']
    if len(keys) != len(set(keys)) or cache['depths'] != list(depths):
        raise ValueError('Expert cache keys or depth contract differ')
    if len(cache['features']) != len(depths) or not depths:
        raise ValueError('Expert cache has the wrong number of feature levels')
    if any(t.shape != (len(keys), 90, 512) or t.dtype != torch.float16 for t in cache['features']):
        raise ValueError('Expert cache must contain FP16 absolute 90-square, 512-wide features')
    if cache['wdl'].shape != (len(keys), 3) or cache['wdl'].dtype != torch.float32:
        raise ValueError('Expert cache WDL dimensions or precision differ')


def extend_cache(base, shards, depths):
    """Preserve the base key order and tensor values, then append disjoint shards."""
    caches = [base, *shards]
    keys, seen = [], set()
    for cache in caches:
        validate_cache(cache, depths)
        if seen.intersection(cache['keys']):
            raise ValueError('Expert feature shards contain overlapping history keys')
        keys.extend(cache['keys'])
        seen.update(cache['keys'])
    return {'keys': keys, 'depths': list(depths),
            'features': [torch.cat([c['features'][i] for c in caches]) for i in range(len(depths))],
            'wdl': torch.cat([c['wdl'] for c in caches])}


def _history_pair(row):
    history = row['history']
    if (len(history) != len(row['moves']) + 1 or history[0] != row['initial_fen'] or
            history[-1] != row['fen'] or history_key(history) != row['feature_key']):
        raise ValueError('Recorded expert context has inconsistent full-history identity')
    # Native replay also changes fullmove counters when the initial side becomes
    # black. Per-FEN mirrors are different full-history keys.
    mirror = replay(mirror_fen(row['initial_fen']), [mirror_move(move) for move in row['moves']])
    return history, mirror


def _history_chunk(rows):
    return [_history_pair(row) for row in rows]


def recorded_histories(path, workers=1):
    """Ordered native color counterparts with bounded CPU preparation workers."""
    if workers < 1:
        raise ValueError('Positive history preparation worker count required')
    def rows():
        with Path(path).open() as handle:
            for line in handle:
                row = json.loads(line)
                yield {k: row[k] for k in ('initial_fen', 'moves', 'history', 'fen', 'feature_key')}
    if workers == 1:
        for row in rows():
            yield from _history_pair(row)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending, chunk = deque(), []
        for row in rows():
            chunk.append(row)
            if len(chunk) == 128:
                pending.append(pool.submit(_history_chunk, chunk))
                chunk = []
                if len(pending) >= 2 * workers:
                    for pair in pending.popleft().result():
                        yield from pair
        if chunk:
            pending.append(pool.submit(_history_chunk, chunk))
        for job in pending:
            for pair in job.result():
                yield from pair


def worker(runtime, shard_index):
    plan_path = runtime / 'plan.json'
    plan = json.loads(plan_path.read_text())
    contexts = runtime / 'new-histories.jsonl'
    if digest(contexts) != plan['new_histories_sha256'] or digest(plan['weights']) != plan['expert_sha256']:
        raise ValueError('Pinned expert or prepared full histories changed')
    count = len(plan['gpu_indices'])
    if not 0 <= shard_index < count:
        raise ValueError('Invalid expert feature shard index')
    histories, keys = [], []
    with contexts.open() as handle:
        for index, line in enumerate(handle):
            if index % count == shard_index:
                row = json.loads(line)
                if history_key(row['history']) != row['feature_key']:
                    raise ValueError('Prepared full-history identity changed')
                keys.append(row['feature_key'])
                histories.append(row['history'])
    torch.set_num_threads(plan['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    expert = FrozenPx0(plan['weights']).cuda()
    depths = plan['depths']
    if (sorted(set(depths)) != depths or depths[0] < 0 or depths[-1] >= expert.depth or
            expert.dim != 512 or sum(p.numel() for p in expert.parameters() if p.requires_grad)):
        raise ValueError('Expert architecture or frozen-depth contract differs')
    levels = [torch.empty((len(keys), 90, 512), dtype=torch.float16) for _ in depths]
    values = torch.empty((len(keys), 3), dtype=torch.float32)
    started = time.monotonic()
    size = plan['batch_size']
    for offset in range(0, len(keys), size):
        batch = torch.stack([encode_history(h) for h in histories[offset:offset + size]]).cuda()
        features, wdl = expert(batch, depths=depths)
        if not all(torch.isfinite(f).all().item() for f in features) or not torch.isfinite(wdl).all().item():
            raise ValueError('Expert produced nonfinite features')
        end = offset + len(batch)
        for level, feature in zip(levels, features):
            level[offset:end].copy_(feature.cpu().half())
        values[offset:end].copy_(wdl.cpu())
        if offset == 0 or end % (size * 64) == 0 or end == len(keys):
            print(json.dumps({'shard': shard_index, 'histories_extracted': end,
                              'total': len(keys), 'seconds': time.monotonic() - started}), flush=True)
    cache = {'keys': keys, 'depths': depths, 'features': levels, 'wdl': values}
    validate_cache(cache, depths)
    output = runtime / f'shard-{shard_index}.pt'
    partial = output.with_suffix('.pt.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Preserve existing feature shard; use a fresh runtime')
    torch.save(cache, partial)
    partial.replace(output)
    summary = {'roots': len(keys), 'shard': shard_index, 'gpu_index': plan['gpu_indices'][shard_index],
               'gpu_name': torch.cuda.get_device_name(0), 'frozen_expert': True,
               'all_features_and_wdl_finite': True, 'depths': depths,
               'seconds': time.monotonic() - started}
    atomic_json(runtime / f'shard-{shard_index}.manifest.json',
                manifest('recorded_expert_feature_shard', plan, [plan_path, contexts, plan['weights']], [output], summary))
    print(json.dumps(summary), flush=True)


def build(args):
    runtime, output = Path(args.runtime), Path(args.output)
    if runtime.exists() or output.exists() or output.with_suffix('.pt.partial').exists():
        raise FileExistsError('Preserve existing expert caches; use fresh output and runtime paths')
    if (not args.gpu_indices or len(set(args.gpu_indices)) != len(args.gpu_indices) or
            any(i < 0 or i >= torch.cuda.device_count() for i in args.gpu_indices) or
            min(args.batch_size, args.cpu_threads, args.history_workers) < 1):
        raise ValueError('Distinct available GPUs and positive batch/thread budgets required')
    proof_path = Path(args.base_proof)
    proof = json.loads(proof_path.read_text())
    cache_path = Path(args.base_cache)
    if proof['status'] != 'complete' or digest(cache_path) != proof['outputs'][str(cache_path)]['sha256']:
        raise ValueError('Base expert cache is incomplete or its bytes changed')
    expert_sha = digest(args.weights)
    if (expert_sha != proof['inputs'][args.weights]['sha256'] or
            expert_sha != proof['verification']['expert_sha256']):
        raise ValueError('Cannot combine expert features from different weights')
    # Weight identity alone is insufficient when the feature formula changes.
    expert_path = Path(__file__).with_name('expert.py')
    if digest(expert_path) != proof['code']['source_sha256']['src/xqgeneral/expert.py']:
        raise ValueError('Base feature encoder implementation differs')
    base = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    depths = proof['verification']['depths']
    validate_cache(base, depths)
    existing, additions = set(base['keys']), {}
    print(json.dumps({'phase': 'native_mirror_history_preparation', 'workers': args.history_workers}), flush=True)
    for index, history in enumerate(recorded_histories(args.roots, args.history_workers)):
        key = history_key(history)
        if key not in existing:
            additions.setdefault(key, history)
        if (index + 1) % 4096 == 0:
            print(json.dumps({'histories_prepared': index + 1, 'new_unique_histories': len(additions)}), flush=True)
    runtime.mkdir(parents=True)
    contexts = runtime / 'new-histories.jsonl'
    write_jsonl(contexts, ({'feature_key': k, 'history': h} for k, h in additions.items()))
    plan = dict(vars(args), depths=depths, expert_sha256=expert_sha, new_histories_sha256=digest(contexts),
                roots_sha256=digest(args.roots), base_cache_sha256=proof['outputs'][str(cache_path)]['sha256'],
                base_roots=len(existing), new_roots=len(additions),
                resume='Preserve interrupted shards and all source inputs; use fresh runtime/output for retry.',
                inspect=f'Read {runtime}/worker-*.log and final feature manifest; course data remain unchanged.')
    atomic_json(runtime / 'plan.json', plan)
    print(json.dumps({'phase': 'feature_extension_plan', 'base_roots': len(existing),
                      'new_roots': len(additions), 'gpu_indices': args.gpu_indices}), flush=True)
    del additions
    children, logs = [], []
    try:
        for index, gpu in enumerate(args.gpu_indices):
            log = (runtime / f'worker-{index}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            children.append(subprocess.Popen([sys.executable, '-u', '-m', 'xqgeneral.extend_features',
                '--worker-runtime', str(runtime), '--shard-index', str(index)], env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError(f'Expert extraction failed; preserved worker exit codes: {codes}')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for log in logs:
            log.close()
    shards, inputs = [], [proof_path, cache_path, args.weights, args.roots, runtime / 'plan.json', contexts]
    for index in range(len(args.gpu_indices)):
        path, report = runtime / f'shard-{index}.pt', runtime / f'shard-{index}.manifest.json'
        shard_proof = json.loads(report.read_text())
        if shard_proof['status'] != 'complete' or digest(path) != shard_proof['outputs'][str(path)]['sha256']:
            raise ValueError('Extracted feature shard changed or is incomplete')
        shards.append(torch.load(path, map_location='cpu', weights_only=True, mmap=True))
        inputs.extend([path, report])
    result = extend_cache(base, shards, depths)
    if len(result['keys']) != len(existing) + plan['new_roots']:
        raise ValueError('Expert cache extension lost a history')
    prepared_keys = set()
    with contexts.open() as handle:
        for line in handle:
            prepared_keys.add(json.loads(line)['feature_key'])
    if set(result['keys']) != existing | prepared_keys:
        raise ValueError('Expert cache extension contains unexpected histories')
    prefix = len(base['keys'])
    if (result['keys'][:prefix] != base['keys'] or
            any(not torch.equal(new[:prefix], old) for new, old in zip(result['features'], base['features'])) or
            not torch.equal(result['wdl'][:prefix], base['wdl'])):
        raise ValueError('Expert extension changed a preserved base feature')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.pt.partial')
    torch.save(result, partial)
    partial.replace(output)
    summary = {'base_roots': prefix, 'new_roots': plan['new_roots'], 'roots': len(result['keys']),
               'depths': depths, 'feature_shapes': [list(t.shape) for t in result['features']],
               'expert_sha256': expert_sha, 'base_keys_features_and_wdl_bitwise_preserved': True,
               'recorded_color_counterparts_included': True, 'dataset_files_modified': False,
               'stored_feature_dtype': 'float16', 'extraction_compute_dtype': 'float32',
               'gpu_indices': args.gpu_indices, 'student_training_executed': False}
    atomic_json(output.with_suffix('.manifest.json'),
                manifest('immutable_extended_expert_feature_cache', vars(args), inputs, [output], summary))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots')
    parser.add_argument('--base-cache', default='data/move-quality-selfplay-v3/features-16.pt')
    parser.add_argument('--base-proof', default='data/move-quality-selfplay-v3/features-16.manifest.json')
    parser.add_argument('--weights', default='models/px0-latest.pb.gz')
    parser.add_argument('--runtime')
    parser.add_argument('--output')
    parser.add_argument('--gpu-indices', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--history-workers', type=int, default=16)
    parser.add_argument('--worker-runtime')
    parser.add_argument('--shard-index', type=int)
    args = parser.parse_args()
    if args.worker_runtime is not None:
        if args.shard_index is None:
            raise ValueError('A feature worker requires its shard index')
        worker(Path(args.worker_runtime), args.shard_index)
    else:
        if any(x is None for x in [args.roots, args.runtime, args.output]):
            parser.error('--roots, --runtime and --output are required')
        build(args)


if __name__ == '__main__':
    main()
