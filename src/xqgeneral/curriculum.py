"""Reproducible four-course runner; each course initializes from the selected best."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .evidence import atomic_json, digest, manifest
from .curriculum_data import STAGES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/research-v1.json')
    parser.add_argument('--mode', choices=['bridge', 'text_lora'], default='bridge')
    args = parser.parse_args()
    recipe = json.loads(Path(args.config).read_text())
    root = Path(recipe['output']) / args.mode / 'curriculum'
    configs = root / 'configs'
    configs.mkdir(parents=True, exist_ok=True)
    results, previous = [], None
    for index, (stage, mixture) in enumerate(zip(STAGES, recipe['course_mixtures'])):
        if stage not in mixture or any(s not in STAGES[:index + 1] for s in mixture):
            raise ValueError('Curriculum mixture must include the new course and only preceding courses')
        config = {key: value for key, value in recipe.items() if key not in
                  {'course_mixtures', 'steps_per_course', 'quality_targets', 'expert_weights'}}
        output = root / stage
        config.update(mode=args.mode, stages=[stage], mixture=mixture, steps=recipe['steps_per_course'],
                      min_steps=recipe['steps_per_course'], output=str(output))
        if previous:
            config['init_from'] = str(previous)
        path = configs / f'{stage}.json'
        if (output / 'metrics.json').exists():
            saved = json.loads((output / 'config.json').read_text())
            if saved != config:
                raise ValueError('Completed curriculum configuration differs; choose a new experiment')
            proof = json.loads((output / 'manifest.json').read_text())
            if digest(output / 'adapter.pt') != proof['outputs'][str(output / 'adapter.pt')]['sha256']:
                raise ValueError('Completed checkpoint hash differs from its manifest')
        else:
            atomic_json(path, config)
            command = [sys.executable, '-m', 'xqgeneral.train', '--config', str(path)]
            if (output / 'latest.pt').exists():
                command += ['--resume', str(output / 'latest.pt')]
            with (root / f'{stage}.log').open('a') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        results.append(json.loads((output / 'metrics.json').read_text()))
        previous = output / 'adapter.pt'
        print(json.dumps({'event': 'course_complete', 'stage': stage, 'metrics': results[-1]}), flush=True)
    atomic_json(root / 'manifest.json', manifest('sequential_curriculum', {**recipe, 'mode': args.mode},
                [args.config], [previous], {'courses': results, 'replay_mixtures_verified': True}))


if __name__ == '__main__':
    main()
