"""Run the unchanged training implementation, then record each worker's resources."""
import argparse
import json
import os
from pathlib import Path

import torch

from .evidence import atomic_json
from . import training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stop-after', required=True, type=int)
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    training.main()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = int(os.environ.get('LOCAL_RANK', 0))
    root = Path(config['output'])
    actual_step = json.loads((root / 'training.jsonl').read_text().splitlines()[-1])['step']
    atomic_json(root / f'resources-step-{actual_step}-rank-{rank}.json', {
        'rank': rank, 'world_size': world, 'actual_step': actual_step,
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
        'total_memory_gib': torch.cuda.get_device_properties(device).total_memory / 2**30,
        'training_main_returned_successfully': True})


if __name__ == '__main__':main()
