"""Train explanations from a completed selected checkpoint with explicit replay."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from .evidence import atomic_json, digest, load_jsonl


def prepare_config(saved, recipe, source_path, output):
    """Compile the same SFT budget and architecture used by the guarded launcher."""
    architecture = ['model_path', 'model_revision', 'mode', 'decoder_bridge_positions', 'bridge_width',
                    'board_tokens', 'expert_feature_depths', 'lora_rank', 'board_text']
    if any(k in recipe and recipe[k] != saved.get(k) for k in architecture):
        raise ValueError('SFT architecture must preserve the selected checkpoint configuration')
    if type(recipe.get('require_clean_foundation_handoff', False)) is not bool:
        raise ValueError('Clean foundation handoff requirement must be a boolean')
    world, batch = recipe.get('ddp_world_size', 1), recipe['batch_size']
    if type(world) is not int or world < 1 or type(batch) is not int or batch < 1 or batch % world:
        raise ValueError('Global SFT batch must be divisible by the positive DDP world size')
    micro = recipe.get('micro_batch_size', batch // world)
    if type(micro) is not int or micro < 1:
        raise ValueError('SFT micro batch must be a positive integer')
    if (type(recipe['epochs']) not in (int, float) or not math.isfinite(recipe['epochs']) or
            recipe['epochs'] <= 0 or type(recipe['max_steps']) is not int or recipe['max_steps'] < 1 or
            type(recipe['min_steps']) is not int or recipe['min_steps'] < 1):
        raise ValueError('SFT epochs and step budgets must be positive finite numbers')
    config = {k: saved[k] for k in architecture if k in saved}
    config.update({k: v for k, v in recipe.items() if k not in {'epochs', 'max_steps'}})
    primary = recipe['stages'][0]
    weight = recipe['mixture'][primary]
    if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError('Primary SFT mixture weight must be positive and at most one')
    count = sum(r['stage'] == primary for r in load_jsonl(Path(recipe['data_path']) / 'train.jsonl'))
    if not count:
        raise ValueError('The primary SFT dataset is empty')
    steps = min(recipe['max_steps'], math.ceil(count * recipe['epochs'] / (batch * weight)))
    config.update(steps=steps, min_steps=min(steps, recipe['min_steps']),
                  init_from=str(source_path), output=str(output))
    return config


def training_command(config, config_path, resume=None):
    """Launch one controller-owned job with the configured number of workers."""
    world = config.get('ddp_world_size', 1)
    if type(world) is not int or world < 1:
        raise ValueError('SFT DDP world size must be a positive integer')
    command = [sys.executable]
    if world > 1:
        command += ['-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                    f'--nproc_per_node={world}', '--module', 'xqgeneral.train']
    else:
        command += ['-m', 'xqgeneral.train']
    command += ['--config', str(config_path)]
    if resume is not None:
        command += ['--resume', str(resume)]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', default='configs/explanation-sft.json')
    parser.add_argument('--init', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Verify the completed parent and write the exact training config without launching workers')
    args = parser.parse_args()
    source_path = Path(args.init)
    parent = source_path.parent
    proof = json.loads((parent / 'manifest.json').read_text())
    if proof['status'] != 'complete' or digest(source_path) != proof['outputs'][str(source_path)]['sha256']:
        raise ValueError('SFT must initialize from a completed, hash-verified selected checkpoint')
    saved = json.loads((parent / 'config.json').read_text())
    if proof.get('kind') == 'raw_qa_gated_foundation_handoff':
        from .foundation_handoff import validate_sft_handoff
        if digest(parent / 'config.json') != proof['outputs'][str(parent / 'config.json')]['sha256']:
            raise ValueError('Completed foundation handoff configuration changed')
        validate_sft_handoff(proof, saved)
    recipe = json.loads(Path(args.recipe).read_text())
    if (recipe.get('require_clean_foundation_handoff') is True and
            proof.get('kind') != 'raw_qa_gated_foundation_handoff'):
        raise ValueError('This SFT recipe requires the completed four-course clean foundation handoff')
    config = prepare_config(saved, recipe, source_path, args.output)
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
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Prepared SFT configuration differs; preserve it and use a fresh output')
    atomic_json(path, config)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'config': str(path), 'steps': config['steps'],
                          'ddp_world_size': config.get('ddp_world_size', 1), 'training_started': False}), flush=True)
        return
    resume = root / 'latest.pt' if (root / 'latest.pt').exists() else None
    command = training_command(config, path, resume)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
