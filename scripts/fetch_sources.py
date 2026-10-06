"""Fetch exact public source revisions without modifying existing checkouts."""
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    for entry in json.loads((root / 'configs/sources.json').read_text()):
        dest = root / 'vendor' / entry['name']
        if dest.exists():
            print(f"Preserving existing source tree: {entry['name']}", flush=True)
            continue
        dest.mkdir(parents=True)
        subprocess.run(['git', 'init', str(dest)], check=True)
        subprocess.run(['git', '-C', str(dest), 'remote', 'add', 'origin', entry['url']], check=True)
        subprocess.run(['git', '-C', str(dest), 'fetch', '--depth', '1', 'origin', entry['revision']], check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', 'FETCH_HEAD'], check=True)


if __name__ == '__main__':
    main()
