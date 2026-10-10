"""Stream completed question histories into bounded immutable Px0 cache shards."""
import argparse
import json
from pathlib import Path
import time

import torch

from .evidence import digest, file_signature, history_key, iter_jsonl
from .expert import FrozenPx0, encode_history
from .feature_store import FeatureStoreWriter


def build(sources, source_proof, weights, depths, output, *, base_cache=None, base_proof=None,
          batch_size=32, shard_rows=512, device='cuda', cpu_threads=4):
    if (any(type(n) is not int or n < 1 for n in [batch_size, shard_rows, cpu_threads]) or
            bool(base_cache) != bool(base_proof) or not sources):
        raise ValueError('Positive extraction budgets, sources and a complete base pair required')
    sources = [Path(p).resolve() for p in sources]
    if len(set(sources)) != len(sources):
        raise ValueError('Feature extraction sources must be distinct ordered files')
    proof_path, weights = Path(source_proof).resolve(), Path(weights).resolve()
    proof = json.loads(proof_path.read_text())
    expected = {str(Path(p).resolve()): b for p, b in proof.get('outputs', {}).items()}
    if proof.get('status') != 'complete' or any(str(p) not in expected for p in sources):
        raise ValueError('Completed question production must bind every supplied source')
    paths = [*sources, proof_path, weights]
    signatures = {str(p): file_signature(p) for p in paths}
    bindings = {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths}
    if any(bindings[str(p)] != expected[str(p)] for p in sources):
        raise ValueError('Completed source question bytes changed')
    if any(file_signature(p) != signatures[str(p)] for p in paths):
        raise ValueError('Feature extraction inputs changed during initial verification')
    started = time.monotonic()
    config = {'weights': str(weights), 'sources': [str(p) for p in sources],
              'source_proof': str(proof_path), 'batch_size': batch_size, 'shard_rows': shard_rows,
              'device': device, 'cpu_threads': cpu_threads,
              'native_question_validation_repeated': False}
    writer = FeatureStoreWriter(output, depths, bindings[str(weights)]['sha256'], config=config)
    if base_cache:
        writer.add_existing(base_cache, base_proof)
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    expert, pending, emitted, scanned = None, [], 0, 0

    def extract():
        nonlocal expert, emitted
        if not pending:
            return
        if expert is None:
            expert = FrozenPx0(str(weights)).to(device)
            if (expert.dim != 512 or sorted(set(depths)) != list(depths) or
                    min(depths) < 0 or max(depths) >= expert.depth or
                    any(p.requires_grad for p in expert.parameters())):
                raise ValueError('Pinned expert architecture, depths or frozen contract differ')
        levels = [torch.empty((len(pending), 90, 512), dtype=torch.float16) for _ in depths]
        wdl = torch.empty((len(pending), 3), dtype=torch.float32)
        with torch.inference_mode():
            for start in range(0, len(pending), batch_size):
                group = pending[start:start + batch_size]
                planes = torch.stack([encode_history(h) for key, h in group]).to(device)
                features, values = expert(planes, depths=depths)
                end = start + len(group)
                if (len(features) != len(depths) or
                        any(not torch.isfinite(f).all().item() for f in [*features, values])):
                    raise ValueError('Pinned expert returned invalid features or WDL')
                for dst, src in zip(levels, features, strict=True):
                    dst[start:end].copy_(src.cpu().half())
                wdl[start:end].copy_(values.cpu())
        writer.append({'keys': [k for k, h in pending], 'depths': list(depths),
                       'features': levels, 'wdl': wdl})
        emitted += len(pending)
        pending.clear()
        print(json.dumps({'phase': 'bounded_feature_extraction', 'new_histories': emitted,
                          'shards': len(writer.shards), 'seconds': time.monotonic() - started}), flush=True)

    seen = set(writer.seen)
    for source in sources:
        for row in iter_jsonl(source):
            if (row.get('split') not in {'train', 'validation', 'test'} or
                    not isinstance(row.get('history'), list) or not row['history'] or
                    row.get('feature_key') != history_key(row['history']) or
                    row.get('fen') != row['history'][-1]):
                raise ValueError('Question split, board or full-history feature identity differs')
            scanned += 1
            key = row['feature_key']
            if key not in seen:
                seen.add(key); pending.append((key, row['history']))
                if len(pending) == shard_rows:
                    extract()
    extract()
    result = writer.finish(source_bindings=bindings, source_signatures=signatures)
    return {'status': 'complete', 'roots': result['roots'], 'base_roots': result['base_roots'],
            'new_roots': emitted, 'question_rows_streamed': scanned, 'shards': len(writer.shards),
            'all_question_histories_bound': True, 'expert_loaded': expert is not None,
            'seconds': time.monotonic() - started, 'student_training_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--source-proof', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--depths', type=int, nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-cache')
    parser.add_argument('--base-proof')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--shard-rows', type=int, default=512)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = build(args.source, args.source_proof, args.weights, args.depths, args.output,
                   base_cache=args.base_cache, base_proof=args.base_proof, batch_size=args.batch_size,
                   shard_rows=args.shard_rows, device=args.device, cpu_threads=args.cpu_threads)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
