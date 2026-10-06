"""Launch a run from a preserved source snapshot, safe across later development."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('Provide an argument-vector command after --')
    root = (ROOT / args.output).resolve()
    source = root / 'source/xqgeneral'
    metadata = root / 'source-manifest.json'
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        actual = {p.name: sha(p) for p in sorted(source.glob('*.py'))}
        if saved['source_sha256'] != actual:
            raise ValueError('Preserved run source was modified')
    else:
        if source.exists():
            raise FileExistsError('Unfinished source snapshot exists; use a fresh run directory')
        source.mkdir(parents=True)
        for file in sorted((ROOT / 'src/xqgeneral').glob('*.py')):
            shutil.copy2(file, source / file.name)
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        saved = {'revision': rev, 'source_sha256': {p.name: sha(p) for p in sorted(source.glob('*.py'))},
                 'purpose': 'immutable execution source for resumable research'}
        metadata.write_text(json.dumps(saved, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=str(root / 'source'), XQGENERAL_CODE_REVISION=saved['revision'])
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
