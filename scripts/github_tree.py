"""Emit committed UTF-8 source tree entries for the connected GitHub Git API.

This helper uses no credentials and performs no network writes. The API caller
must read and verify the resulting tree/commit before moving a branch ref.
"""
import json
import subprocess


def main():
    raw = subprocess.check_output(['git', 'ls-tree', '-rz', 'HEAD'])
    entries = []
    for line in raw.split(b'\0'):
        if not line:
            continue
        meta, path = line.split(b'\t', 1)
        mode, kind, sha = meta.decode().split()
        if kind != 'blob':
            raise ValueError('Source-only tree cannot contain a submodule')
        data = subprocess.check_output(['git', 'cat-file', 'blob', sha])
        entries.append({'path': path.decode(), 'mode': mode, 'type': 'blob', 'content': data.decode('utf-8')})
    print(json.dumps(entries))


if __name__ == '__main__':
    main()
