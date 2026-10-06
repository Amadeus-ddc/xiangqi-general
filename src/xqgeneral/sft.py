"""Train explanations from a completed selected checkpoint with explicit replay."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from .evidence import atomic_json, digest, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', default='configs/explanation-sft.json')
    parser.add_argument('--init', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source_path = Path(args.init)
    parent = source_path.parent
    proof = json.loads((parent / 'manifest.json').read_text())
    if proof['status'] != 'complete' or digest(source_path) != proof['outputs'][str(source_path)]['sha256']:
        raise ValueError('SFT must initialize from a completed, hash-verified selected checkpoint')
    saved = json.loads((parent / 'config.json').read_text())
    recipe = json.loads(Path(args.recipe).read_text())
    architecture = ['model_path', 'model_revision', 'mode', 'decoder_bridge_positions', 'bridge_width',
                    'board_tokens', 'expert_feature_depths', 'lora_rank', 'board_text']
    config = {k: saved[k] for k in architecture if k in saved}
    config.update({k: v for k, v in recipe.items() if k not in {'epochs', 'max_steps'}})
    primary = recipe['stages'][0]
    count = sum(r['stage'] == primary for r in load_jsonl(Path(recipe['data_path']) / 'train.jsonl'))
    if not count:
        raise ValueError('The primary SFT dataset is empty')
    steps = min(recipe['max_steps'], math.ceil(count * recipe['epochs'] /
                (recipe['batch_size'] * recipe['mixture'][primary])))
    config.update(steps=steps, min_steps=min(steps, recipe['min_steps']), init_from=str(source_path), output=args.output)
    root = Path(args.output)
    if (root / 'metrics.json').exists():
        if json.loads((root / 'config.json').read_text()) != config:
            raise ValueError('Completed SFT configuration differs')
        selected = json.loads((root / 'manifest.json').read_text())
        if digest(root / 'adapter.pt') != selected['outputs'][str(root / 'adapter.pt')]['sha256']:
            raise ValueError('Completed SFT checkpoint changed')
        print((root / 'metrics.json').read_text(), flush=True)
        return
    path = root.parent / (root.name + '.config.json')
    atomic_json(path, config)
    command = [sys.executable, '-m', 'xqgeneral.train', '--config', str(path)]
    if (root / 'latest.pt').exists():
        command += ['--resume', str(root / 'latest.pt')]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
