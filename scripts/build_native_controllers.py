"""Build a weight-free controller reference without editing or downloading vendors."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--vendor', type=Path, default=root / 'vendor/pikafish')
    parser.add_argument('--output', type=Path, default=root / 'build/native-controllers')
    args = parser.parse_args()
    pinned = next(item['revision'] for item in json.loads((root / 'configs/sources.json').read_text())
                  if item['name'] == 'pikafish')
    revision = subprocess.run(['git', '-C', str(args.vendor), 'rev-parse', 'HEAD'],
                              check=True, text=True, capture_output=True).stdout.strip()
    if revision != pinned:
        raise ValueError('Controller reference requires the pinned Pikafish revision')
    subprocess.run(['git', '-C', str(args.vendor), 'diff', '--quiet', 'HEAD', '--', 'src'], check=True)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [root / 'scripts/native_controllers.cc', *[args.vendor / 'src' / path for path in
               ['position.cpp', 'attacks.cpp', 'bitboard.cpp', 'nnue/features/half_ka_v2_hm.cpp']]]
    command = ['g++', '-std=c++20', '-O2', '-DNDEBUG', '-DIS_64BIT', '-ffunction-sections', '-fdata-sections',
               '-I' + str(args.vendor / 'src'), *map(str, sources), '-Wl,--gc-sections', '-pthread',
               '-o', str(args.output / 'controllers')]
    log = args.output / ('build-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.log')
    with log.open('x') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Native controller build failed; inspect {log}')
    proof = {'status': 'complete', 'vendor_revision': pinned, 'command': command,
             'executable_sha256': hashlib.sha256((args.output / 'controllers').read_bytes()).hexdigest(),
             'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [*sources, *sorted((args.vendor / 'src').rglob('*.h'))]},
             'weights_loaded': False}
    (args.output / 'manifest.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'native_controller_reference_built': True, 'vendor_revision': pinned,
                      'weights_loaded': False}))


if __name__ == '__main__':
    main()
