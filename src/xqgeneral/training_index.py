"""Bound JSONL offsets and compact metadata; training reads only requested rows."""
import argparse
import hashlib
import json
import mmap
import operator
import os
from pathlib import Path
import sqlite3
import struct

import numpy as np

from .course_tasks import row_profile
from .evidence import atomic_json, digest, file_signature, manifest

INDEX_PROFILE = 'indexed_training_jsonl_v1'
ARTIFACTS = ('records.bin', 'identities.bin', 'metadata.json', 'manifest.json')
FIELDS = [('source', '<u4'), ('offset', '<u8'), ('bytes', '<u8'), ('stage', '<u2'),
          ('profile', '<u2'), ('id_offset', '<u8'), ('id_bytes', '<u4'), ('feature', '<u4')]
DTYPE = np.dtype(FIELDS)
PACK = struct.Struct('<IQQHHQII')


def index_paths(directory):
    return [Path(directory) / name for name in ARTIFACTS]


def training_row_indexes(config):
    if 'row_index_paths' not in config:
        return {}
    from .finite_training import FINITE_PROFILE, data_profile
    roots = config['row_index_paths']
    if (not isinstance(roots, dict) or 'train' not in roots or
            set(roots) - {'train', 'validation'} or
            any(not isinstance(p, str) or not p for p in roots.values()) or
            data_profile(config) != FINITE_PROFILE):
        raise ValueError('Row indexes require train/optional validation paths and finite traversal')
    return {split: Path(path) for split, path in roots.items()}


def _identity(row, split):
    if (not isinstance(row, dict) or row.get('split', split) != split or
            any(not isinstance(row.get(k), str) or not row[k] for k in ['stage', 'id', 'feature_key'])):
        raise ValueError('Indexed rows need the declared split and nonempty stage, ID and feature key')
    return row['stage'], row_profile(row), row['id'], row['feature_key']


def build_index(paths, output, *, split='train'):
    """Stream source bytes once; publish a complete manifest only after all guards."""
    if split not in {'train', 'validation'}:
        raise ValueError('Only training and validation corpora may become training row indexes')
    sources = [Path(p).resolve() for p in paths]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError('Index sources must be a nonempty ordered list of distinct files')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stages, profiles, features = {}, {}, {}
    signatures = {p: file_signature(p) for p in sources}
    inputs, counts, total = {}, {}, 0
    atomic_json(output / 'progress.json', {'status': 'building', 'split': split, 'rows': 0})
    db = sqlite3.connect(output / 'identity-check.sqlite')
    try:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA cache_size=-262144')
        db.execute('CREATE TABLE ids(identity BLOB PRIMARY KEY) WITHOUT ROWID')
        with (output / 'records.bin').open('xb') as records, (output / 'identities.bin').open('xb') as identities:
            for source_index, path in enumerate(sources):
                hasher = hashlib.sha256()
                with path.open('rb') as handle:
                    while True:
                        offset = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        hasher.update(line)
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        stage, profile, identity, feature = _identity(row, split)
                        encoded = identity.encode('utf-8')
                        try:
                            db.execute('INSERT INTO ids VALUES (?)', (encoded,))
                        except sqlite3.IntegrityError as error:
                            raise ValueError('Training row index contains a duplicate ID') from error
                        for table, value, capacity in [(stages, stage, 2**16), (profiles, profile, 2**16),
                                                       (features, feature, 2**32)]:
                            if value not in table:
                                if len(table) >= capacity:
                                    raise ValueError('Training row metadata table capacity exceeded')
                                table[value] = len(table)
                        if len(encoded) >= 2**32:
                            raise ValueError('Training row ID is too large')
                        records.write(PACK.pack(source_index, offset, len(line), stages[stage], profiles[profile],
                                                identities.tell(), len(encoded), features[feature]))
                        identities.write(encoded)
                        counts[stage] = counts.get(stage, 0) + 1
                        total += 1
                        if total % 100000 == 0:
                            atomic_json(output / 'progress.json', {'status': 'building', 'split': split,
                                                                  'rows': total, 'source_index': source_index})
                if file_signature(path) != signatures[path]:
                    raise ValueError('Source changed while building its row index')
                inputs[str(path)] = {'sha256': hasher.hexdigest(), 'bytes': path.stat().st_size}
        db.commit()
        if any(file_signature(p) != signatures[p] for p in sources):
            raise ValueError('A source changed before row index publication')
        metadata = {'profile': INDEX_PROFILE, 'split': split, 'rows': total, 'record_bytes': DTYPE.itemsize,
                    'sources': [str(p) for p in sources], 'stages': list(stages), 'profiles': list(profiles),
                    'features': list(features), 'rows_by_stage': counts,
                    'source_signatures': {str(p): list(s) for p, s in signatures.items()}}
        atomic_json(output / 'metadata.json', metadata)
        proof = {'status': 'complete', 'profile': INDEX_PROFILE, 'split': split, 'rows': total,
                 'rows_by_stage': counts, 'all_row_ids_unique': True,
                 'all_source_bytes_streamed_and_hashed': True, 'independent_test_used': False}
        receipt = manifest('training_jsonl_row_index', {'profile': INDEX_PROFILE, 'split': split},
                           outputs=index_paths(output)[:3], verification=proof)
        receipt['inputs'] = inputs
        receipt['verification']['artifact_signatures'] = {
            str(p): list(file_signature(p)) for p in index_paths(output)[:3]}
        atomic_json(output / 'manifest.json', receipt)
        atomic_json(output / 'progress.json', {'status': 'complete', 'split': split, 'rows': total})
        return proof
    finally:
        db.close()


class IndexedTrainingRows:
    """Read-only sequence in original filtered file order, with metadata-only scans."""
    def __init__(self, directory, sources, stages, *, split='train', verified_hashes=None):
        self.directory = Path(directory).resolve()
        source_paths = [Path(p).resolve() for p in sources]
        signatures = {p: file_signature(p) for p in [*source_paths, *index_paths(self.directory)]}
        hashes = None if verified_hashes is None else {str(Path(p).resolve()): h for p, h in verified_hashes.items()}
        manifest_path = self.directory / 'manifest.json'
        if hashes is not None and hashes.get(str(manifest_path)) != digest(manifest_path):
            raise ValueError('Training row index manifest has changed or is unverified')
        receipt = json.loads((self.directory / 'manifest.json').read_text())
        if (receipt.get('status') != 'complete' or receipt.get('kind') != 'training_jsonl_row_index' or
                receipt.get('config') != {'profile': INDEX_PROFILE, 'split': split}):
            raise ValueError('Training row index is incomplete or has a different profile/split')
        wanted_outputs = {str(p) for p in index_paths(self.directory)[:3]}
        if set(receipt['outputs']) != wanted_outputs or set(receipt['inputs']) != {str(p) for p in source_paths}:
            raise ValueError('Training row index source/artifact paths differ')
        for section in ['inputs', 'outputs']:
            for name, expected in receipt[section].items():
                path = Path(name)
                actual = digest(path) if hashes is None else hashes.get(name)
                if (hashes is not None and section == 'outputs' and
                        list(signatures[path]) != receipt['verification'].get('artifact_signatures', {}).get(name)):
                    actual = digest(path)
                if actual != expected['sha256'] or path.stat().st_size != expected['bytes']:
                    raise ValueError('Training row index has changed or unverified source/artifact bytes')
        self.meta = json.loads((self.directory / 'metadata.json').read_text())
        m = self.meta
        # A caller's already-computed hashes are valid only for unchanged source bytes.
        # A changed filesystem signature gets a fresh full hash, including same-size edits.
        for path in source_paths:
            if list(signatures[path]) != m.get('source_signatures', {}).get(str(path)):
                if digest(path) != receipt['inputs'][str(path)]['sha256']:
                    raise ValueError('Training row source changed since index construction')
        if (m.get('profile') != INDEX_PROFILE or m.get('split') != split or
                m.get('sources') != [str(p) for p in source_paths] or
                type(m.get('rows')) is not int or m['rows'] < 0 or m.get('record_bytes') != DTYPE.itemsize or
                receipt['verification'].get('rows') != m['rows'] or
                receipt['verification'].get('all_row_ids_unique') is not True):
            raise ValueError('Training row index metadata contract differs')
        if any(not isinstance(m.get(name), list) or
               any(not isinstance(v, str) or not v for v in m[name]) or len(set(m[name])) != len(m[name])
               for name in ['stages', 'profiles', 'features']):
            raise ValueError('Training row metadata tables are invalid')
        if (self.directory / 'records.bin').stat().st_size != m['rows'] * DTYPE.itemsize:
            raise ValueError('Training row index offset file is truncated or has extra rows')
        self.records = (np.memmap(self.directory / 'records.bin', mode='r', dtype=DTYPE, shape=(m['rows'],))
                        if m['rows'] else np.empty(0, dtype=DTYPE))
        self._identity_handle = (self.directory / 'identities.bin').open('rb')
        self._identities = (mmap.mmap(self._identity_handle.fileno(), 0, access=mmap.ACCESS_READ)
                            if (self.directory / 'identities.bin').stat().st_size else b'')
        if m['rows'] and (np.any(self.records['source'] >= len(source_paths)) or
                         np.any(self.records['stage'] >= len(m['stages'])) or
                         np.any(self.records['profile'] >= len(m['profiles'])) or
                         np.any(self.records['feature'] >= len(m['features'])) or
                         np.any(self.records['id_bytes'] == 0) or
                         np.any(self.records['id_bytes'] > len(self._identities)) or
                         np.any(self.records['id_offset'] > len(self._identities) - self.records['id_bytes'])):
            raise ValueError('Training row index contains invalid metadata offsets')
        for source, path in enumerate(source_paths):
            size = signatures[path][2]
            for start in range(0, len(self.records), 65536):
                block = self.records[start:start + 65536]
                rows = block[block['source'] == source]
                if np.any(rows['bytes'] == 0) or np.any(rows['bytes'] > size) or np.any(rows['offset'] > size - rows['bytes']):
                    raise ValueError('Training row index contains invalid source offsets')
        allowed = [i for i, stage in enumerate(m['stages']) if stage in stages]
        self.positions = None if len(allowed) == len(m['stages']) else np.flatnonzero(np.isin(self.records['stage'], allowed))
        self.sources, self.split = source_paths, split
        self._signatures = signatures
        self._fds = {}
        self.check_unchanged()

    def __len__(self):
        return len(self.records) if self.positions is None else len(self.positions)

    def _record(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('Training row index is out of range')
        return self.records[index if self.positions is None else self.positions[index]]

    def row_metadata(self, index):
        r = self._record(index)
        start, size = int(r['id_offset']), int(r['id_bytes'])
        return {'id': self._identities[start:start + size].decode('utf-8'),
                'stage': self.meta['stages'][int(r['stage'])], 'split': self.split,
                'task_profile': self.meta['profiles'][int(r['profile'])],
                'feature_key': self.meta['features'][int(r['feature'])]}

    def iter_metadata(self):
        for index in range(len(self)):
            yield self.row_metadata(index)

    def check_unchanged(self):
        if any(file_signature(p) != expected for p, expected in self._signatures.items()):
            raise ValueError('Training source or row index changed after verification')

    def get_batch(self, indices):
        self.check_unchanged()
        result = []
        for index in indices:
            r = self._record(index)
            source = int(r['source'])
            if source not in self._fds:
                self._fds[source] = os.open(self.sources[source], os.O_RDONLY)
            raw = os.pread(self._fds[source], int(r['bytes']), int(r['offset']))
            if len(raw) != int(r['bytes']):
                raise ValueError('Training row source was truncated during lookup')
            row = json.loads(raw)
            m = self.row_metadata(index)
            if _identity(row, self.split) != (m['stage'], m['task_profile'], m['id'], m['feature_key']):
                raise ValueError('Training source row differs from indexed metadata')
            result.append(row)
        self.check_unchanged()
        return result

    def __getitem__(self, index):
        return self.get_batch(range(*index.indices(len(self)))) if isinstance(index, slice) else self.get_batch([index])[0]

    def close(self):
        for fd in getattr(self, '_fds', {}).values():
            os.close(fd)
        self._fds = {}
        if isinstance(getattr(self, '_identities', None), mmap.mmap):
            self._identities.close()
        if hasattr(self, '_identity_handle'):
            self._identity_handle.close()

    def __del__(self):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--split', choices=['train', 'validation'], default='train')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build_index(args.source, args.output, split=args.split), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
