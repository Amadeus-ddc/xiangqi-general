"""Read back actual optimizer checkpoints and fail on any continuation drift."""
import argparse
import json
from pathlib import Path
import torch


def identical(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(identical(v, w) for v, w in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', required=True)
    parser.add_argument('--resumed', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    roots = [Path(args.continuous), Path(args.resumed)]
    a, b = [torch.load(p / 'latest.pt', map_location='cpu', weights_only=True, mmap=True) for p in roots]
    failures = []
    fields = ['step', 'world_size', 'trainable', 'optimizer', 'random_state', 'torch_rng_state',
              'cuda_rng_states', 'tokens_seen', 'initial_loss', 'input_hashes', 'code',
              'compute_contract', 'patience_count', 'best_step', 'best_loss']
    for field in fields:
        if not identical(a[field], b[field]):
            failures.append(field)
    logs = [[{k: v for k, v in json.loads(line).items() if k != 'seconds'}
             for line in (p / 'training.jsonl').read_text().splitlines()] for p in roots]
    if logs[0] != logs[1]:
        failures.append('step_logs')
    differences = [float((value - b['trainable'][name]).abs().max())
                   for name, value in a['trainable'].items()
                   if not torch.equal(value, b['trainable'][name])]
    print(json.dumps({'status': 'failed' if failures else 'complete', 'mismatched_fields': failures,
                      'mismatched_trainable_tensors': len(differences),
                      'max_parameter_absolute_error': max(differences, default=0),
                      'steps': [a['step'], b['step']], 'world_sizes': [a['world_size'], b['world_size']]}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
