"""Verify reusable recorded-source catalogs before opening them read-only.

Complete history/isolation proofs are inherited only for their unchanged input
bytes. Every native target class is recomputed; catalog rows are never evidence
for their own correctness. Generated terminal extensions use the normal builder.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import json
import multiprocessing
from pathlib import Path
import sqlite3

from .course_position_sampling import CLASS_BITS, PositionCatalog, SPLITS
from .course_source_semantics import native_classes
from .evidence import atomic_json, digest, file_signature, manifest, position_key
from .rules import in_check, legal_moves, replay
from .symmetry import mirror_fen

KIND = 'verified_recorded_native_position_catalog'
SOURCE_KIND = 'recorded_full_native_course_root_pools'
HISTORY_KIND = 'recorded_full_native_course_root_readback'
HISTORY_FLAGS = ('complete_source_histories_and_all_native_prefix_endings_recomputed',
                 'prior_root_future_witness_color_and_all_source_history_ownership_reserved',
                 'prior_structured_explanation_pvs_and_branches_reserved',
                 'prior_contexts_may_be_reused_only_in_their_existing_split')


def _paths(values):
    return [str(Path(p).resolve()) for p in values]


def _merge_bindings(bindings, values):
    if not isinstance(values, dict):
        raise ValueError('Completed history artifact bindings are required')
    for name, item in values.items():
        name = str(Path(name).resolve())
        if (not isinstance(item, dict) or type(item.get('bytes')) is not int or item['bytes'] < 0 or
                not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64):
            raise ValueError('Invalid completed history artifact binding')
        if name in bindings and bindings[name] != item:
            raise ValueError('Conflicting completed history artifact bindings')
        bindings[name] = item


def _proof(path, kind):
    signature = file_signature(path)
    value = json.loads(Path(path).read_text())
    if value.get('status') != 'complete' or value.get('kind') != kind:
        raise ValueError('A completed recorded history proof is required')
    item = {'sha256': digest(path), 'bytes': Path(path).stat().st_size}
    if file_signature(path) != signature:
        raise ValueError('Completed history proof changed while being read')
    return value, item


def _source_contract(roots, source_manifests, history_readbacks, reference_data):
    if not roots or len(roots) != len(source_manifests) or len(roots) != len(history_readbacks):
        raise ValueError('Every recorded source needs its complete history readback')
    if len(set(_paths(roots))) != len(roots):
        raise ValueError('Duplicate recorded root source')
    bindings, counts, allowed_references = {}, Counter(), set()
    for root_path, proof_path, history_path in zip(roots, source_manifests, history_readbacks, strict=True):
        source, source_item = _proof(proof_path, SOURCE_KIND)
        history, history_item = _proof(history_path, HISTORY_KIND)
        flags, checked = source.get('verification', {}), history.get('verification', {})
        if (any(flags.get(k) is not True for k in HISTORY_FLAGS) or
                flags.get('source_history_rewritten_or_missing_past_invented') is not False or
                checked.get('all_source_input_and_pool_bytes_checked') is not True):
            raise ValueError('Complete native history and split isolation proof is required')
        declared = flags.get('roots_by_split')
        if (not isinstance(declared, dict) or set(declared) != set(SPLITS) or
                any(type(n) is not int or n < 0 for n in declared.values()) or
                checked.get('roots_and_exclusions_regenerated', {}).get('root') != sum(declared.values())):
            raise ValueError('Completed history readback does not cover all recorded roots')
        source_bindings, history_bindings = {}, {}
        _merge_bindings(source_bindings, source.get('inputs'))
        _merge_bindings(source_bindings, source.get('outputs'))
        _merge_bindings(history_bindings, history.get('inputs'))
        root_name, proof_name = _paths([root_path, proof_path])
        outputs = {str(Path(p).resolve()): v for p, v in source['outputs'].items()}
        if (root_name not in outputs or history_bindings.get(proof_name) != source_item or
                any(history_bindings.get(p) != item for p, item in source_bindings.items())):
            raise ValueError('Complete history readback is not bound to this source and every original input')
        config = source.get('config', {})
        allowed_references.update(_paths(config.get('reference_data', [])))
        if config.get('owner_manifest'):
            allowed_references.add(str(Path(config['owner_manifest']).resolve().parent))
        _merge_bindings(bindings, source_bindings)
        _merge_bindings(bindings, history_bindings)
        _merge_bindings(bindings, history.get('outputs'))
        _merge_bindings(bindings, {str(proof_path): source_item, str(history_path): history_item})
        counts.update(declared)
    for directory in reference_data:
        data = Path(directory).resolve()
        if str(data) not in allowed_references or any(str(data / name) not in bindings for name in
                ('manifest.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl')):
            raise ValueError('Prior reference corpus is not covered by the completed history isolation proof')
    return bindings, counts


def _check_bindings(bindings):
    signatures = {}
    for name, item in bindings.items():
        before = file_signature(name)
        if before[2] != item['bytes'] or digest(name) != item['sha256'] or file_signature(name) != before:
            raise ValueError('Completed catalog or history input bytes changed')
        signatures[name] = before
    return signatures


def _unchanged(signatures):
    if any(file_signature(p) != signature for p, signature in signatures.items()):
        raise ValueError('Catalog or completed history proof changed during verification')


def _open_catalog(path):
    path = Path(path).resolve()
    if not path.is_file() or any(Path(str(path) + suffix).exists() for suffix in ('-wal', '-journal')):
        raise ValueError('Catalog must be a completed standalone SQLite file')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    expected = {'roots': ('id', 'feature', 'split', 'query_key', 'color_key', 'value'),
                'targets': ('root', 'horizon', 'mask', 'variant_move', 'descendant_feature'),
                'game_owners': ('game', 'split'), 'positions': ('position', 'split')}
    try:
        if any(tuple(row[1] for row in db.execute(f'PRAGMA table_info({table})')) != columns
               for table, columns in expected.items()):
            raise ValueError('Catalog schema differs from the native position sampler')
        for index, columns in [('target_order', ('root', 'horizon')), ('split_order', ('split', 'id'))]:
            if tuple(row[2] for row in db.execute(f'PRAGMA index_info({index})')) != columns:
                raise ValueError('Catalog lacks its completed ordered sampling indexes')
    except BaseException:
        db.close()
        raise
    return db


def _recorded_targets(root):
    """Caller must first bind a complete history readback to these exact bytes."""
    if root.get('extension_is_recorded_source_move') is False or root.get('paper_generated_terminal_witness'):
        raise ValueError('Recorded catalog audit does not accept generated terminal extensions')
    future = root.get('future_moves', [])
    witness = root.get('paper_recorded_mate_witness', [])
    if (not isinstance(future, list) or len(future) > 8 or not isinstance(witness, list) or
            len(witness) > 14 or witness and future != witness[:len(future)]):
        raise ValueError('Recorded witness differs from the locked actual future')
    distance = None
    if witness:
        terminal = replay(root['fen'], witness)[-1]
        if not in_check(terminal) or legal_moves(terminal):
            raise ValueError('Recorded witness does not end in native checkmate')
        distance = len(witness)
    elif in_check(root['fen']) and not legal_moves(root['fen']):
        distance = 0
    targets = []
    for horizon, fen in enumerate(replay(root['fen'], future)):
        classes = native_classes(fen, None if distance is None else distance - horizon)
        targets.append((horizon, sum(CLASS_BITS[label] for label in classes)))
    return targets


def _target_batch(roots):
    return [_recorded_targets(root) for root in roots]


def _ordered_targets(roots, workers, batch_size):
    source = iter(roots)
    def batches():
        while batch := list(islice(source, batch_size)):
            yield batch
    if workers == 1:
        for batch in batches():
            yield from zip(batch, _target_batch(batch), strict=True)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for batch in batches():
            pending.append((batch, pool.submit(_target_batch, batch)))
            if len(pending) >= workers * 2:
                values, job = pending.popleft()
                yield from zip(values, job.result(), strict=True)
        for values, job in pending:
            yield from zip(values, job.result(), strict=True)


def verify_recorded_catalog(catalog, roots, source_manifests, history_readbacks, output,
                            reference_data=(), workers=1, batch_size=1):
    """Compare every source JSON/header and target mask, with no sampling cap."""
    output = Path(output)
    if output.exists() or type(workers) is not int or workers < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Preserve previous catalog readbacks; use fresh output and positive workers')
    bindings, expected_counts = _source_contract(roots, source_manifests, history_readbacks, reference_data)
    catalog = Path(catalog).resolve()
    catalog_signature = file_signature(catalog)
    catalog_item = {'sha256': digest(catalog), 'bytes': catalog.stat().st_size}
    if file_signature(catalog) != catalog_signature:
        raise ValueError('Catalog changed before native readback')
    _merge_bindings(bindings, {str(catalog): catalog_item})
    signatures = _check_bindings({p: item for p, item in bindings.items() if p != str(catalog)})
    signatures[str(catalog)] = catalog_signature
    output.mkdir(parents=True)
    db = _open_catalog(catalog)
    root_rows = iter(db.execute('SELECT id,feature,split,query_key,color_key,value FROM roots ORDER BY id'))
    target_rows = iter(db.execute('SELECT root,horizon,mask,variant_move,descendant_feature FROM targets ORDER BY root,horizon'))
    counts, games, checked, targets_checked = Counter(), {}, 0, 0
    def sources():
        for path in roots:
            raw_digest = hashlib.sha256()
            with Path(path).open('rb') as handle:
                for raw in handle:
                    raw_digest.update(raw)
                    if not raw.strip():
                        continue
                    root = json.loads(raw)
                    row = next(root_rows, None)
                    if row is None or row[0] != sum(counts.values()) + 1:
                        raise ValueError('Catalog root order or count differs from the recorded source')
                    split, game = root['split'], root['game_id']
                    if split not in SPLITS or not isinstance(game, str) or not game or games.setdefault(game, split) != split:
                        raise ValueError('Catalog source game split ownership differs')
                    expected = (root['feature_key'], split, position_key(root['fen']),
                                position_key(mirror_fen(root['fen'])), json.dumps(root, ensure_ascii=False))
                    if row[1:] != expected:
                        raise ValueError('Catalog root JSON or sampling header differs from its recorded source')
                    counts[split] += 1
                    yield root
            if raw_digest.hexdigest() != bindings[str(Path(path).resolve())]['sha256']:
                raise ValueError('Recorded source bytes changed during catalog readback')
    try:
        for _, targets in _ordered_targets(sources(), workers, batch_size):
            checked += 1
            for horizon, mask in targets:
                if next(target_rows, None) != (checked, horizon, mask, None, None):
                    raise ValueError('Catalog native target classes or horizons differ from recorded source')
                targets_checked += 1
            if checked % 8192 == 0:
                progress = {'status': 'verifying_native_targets', 'roots_checked': checked,
                            'total_roots': sum(expected_counts.values()), 'targets_checked': targets_checked}
                atomic_json(output / 'state.json', progress)
                print(json.dumps(progress), flush=True)
        if (next(root_rows, None) is not None or next(target_rows, None) is not None or counts != expected_counts or
                dict(db.execute('SELECT game,split FROM game_owners')) != games):
            raise ValueError('Catalog has extra, missing or differently owned source rows')
    finally:
        db.close()
    _unchanged(signatures)
    result = {'status': 'complete', 'roots_checked': checked, 'roots_by_split': dict(counts),
              'targets_checked': targets_checked, 'source_games_checked': len(games),
              'all_recorded_source_json_and_sampling_headers_compared': True,
              'all_native_target_masks_recomputed': True, 'source_full_history_proof_inherited': True,
              'source_history_replayed_again': False, 'generated_terminal_variants_supported': False,
              'unused_positions_table_recomputed': False, 'selected_questions_require_full_native_history_checks': True,
              'model_loaded': False, 'student_training_executed': False, 'student_strength_measured': False}
    atomic_json(output / 'verification.json', result)
    config = {'catalog': str(catalog), 'roots': _paths(roots), 'source_manifests': _paths(source_manifests),
              'history_readbacks': _paths(history_readbacks), 'reference_data': _paths(reference_data),
              'output': str(output.resolve()), 'workers': workers, 'batch_size': batch_size}
    completed = manifest(KIND, config, (), [output / 'verification.json'], result)
    completed['inputs'] = bindings
    atomic_json(output / 'manifest.json', completed)
    atomic_json(output / 'state.json', result)
    return result


def load_catalog(proof_path, *, roots, source_manifests, reference_data):
    """Open only a complete unchanged catalog for the exact declared sources."""
    proof, proof_item = _proof(proof_path, KIND)
    config, flags = proof.get('config', {}), proof.get('verification', {})
    if (any(config.get(k) != _paths(v) for k, v in [('roots', roots), ('source_manifests', source_manifests),
                                                   ('reference_data', reference_data)]) or
            any(flags.get(k) is not True for k in ('all_recorded_source_json_and_sampling_headers_compared',
                                                 'all_native_target_masks_recomputed', 'source_full_history_proof_inherited'))):
        raise ValueError('Completed catalog does not cover the declared sources and reference isolation')
    bindings = {}
    _merge_bindings(bindings, proof.get('inputs'))
    _merge_bindings(bindings, proof.get('outputs'))
    required = [config['catalog'], *_paths(roots), *_paths(source_manifests), *config['history_readbacks']]
    if any(str(Path(p).resolve()) not in bindings for p in required):
        raise ValueError('Completed catalog lacks required source bindings')
    _merge_bindings(bindings, {str(proof_path): proof_item})
    signatures = _check_bindings(bindings)
    _unchanged(signatures)
    value = PositionCatalog.__new__(PositionCatalog)
    value.db = _open_catalog(config['catalog'])
    try:
        _unchanged(signatures)
    except BaseException:
        value.db.close()
        raise
    value.counts = Counter(flags['roots_by_split'])
    value.prior_games, value.generated_descendants = {}, []
    value.verified_input_signatures = signatures
    value.verified_input_bindings = bindings
    return value


def check_catalog_unchanged(catalog):
    """Guard all inherited evidence after a long selection/rendering pass."""
    _unchanged(catalog.verified_input_signatures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--source-manifests', nargs='+', required=True)
    parser.add_argument('--history-readbacks', nargs='+', required=True)
    parser.add_argument('--reference-data', nargs='+', default=[])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(verify_recorded_catalog(args.catalog, args.roots, args.source_manifests,
          args.history_readbacks, args.output, args.reference_data, args.workers, args.batch_size)), flush=True)


if __name__ == '__main__':
    main()
