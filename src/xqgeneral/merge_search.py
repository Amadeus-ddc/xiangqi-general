"""Qualify immutable search collections and merge complete history contexts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from .collect_search import (EVALUATION_FILTER_PROFILE, rebuild_labels, mining_reservations,
                             search_evaluation_filter)
from .course_tasks import validate_context
from .curriculum_data import verify_splits
from .evidence import atomic_json, iter_jsonl, manifest, write_jsonl
from .explanations import (IMMEDIATE_MATE_CONSISTENCY_PROFILE, continuation_positions,
                          immediate_mate_score_consistency, parse_explanation,
                          validate_explanation)
from .prose_moves import validate_prose_moves
from .recorded_search_inputs import BoundInputs
from .search_critical import checked_acceptance_evidence, checked_acceptance_policy
from .search_distillation import reserved_positions
from .search_prose_exclusions import SUMMARY_FIELDS, exclude_reviewed_prose

MERGED_KIND = 'merged_search_consolidated_dataset'


SHARED_SOURCE_PROFILE = 'shared_canonical_sources_v1'


def _checked_source_profile(value):
    if value not in {None,SHARED_SOURCE_PROFILE}:
        raise ValueError('Unknown explicit search-source merge profile')
    return value


def _leaf_dataset(data, source, bound, minimum_labels=32, *, source_profile=None):
    _checked_source_profile(source_profile)
    data = Path(data)
    proof = bound.json(data / 'manifest.json')
    result = proof.get('verification', {})
    student_pool = proof.get('config',{}).get('student_game_inputs') or result.get('student_game_inputs')
    is_student = bool(student_pool)
    if is_student and source_profile!=SHARED_SOURCE_PROFILE:
        raise ValueError('Actual neural-game collections require the explicit shared canonical source profile')
    isolation_flag = ('student_game_mining_isolation_rechecked_during_collection' if is_student else
                      'recorded_mining_isolation_rechecked_during_collection')
    if (proof.get('status') != 'complete' or proof.get('kind') != 'search_consolidated_dataset' or
            type(result.get('accepted_training_roots')) is not int or result['accepted_training_roots'] < minimum_labels or
            any(result.get(k) is not True for k in ['neural_consolidation_executed',
                'structured_facts_verified', 'explicit_prose_moves_natively_verified',
                'all_branch_positions_isolated', 'full_history_termination_checked',
                isolation_flag,
                'selected_branch_evaluation_filter_applied']) or
            result.get('evaluation_filter_profile') != EVALUATION_FILTER_PROFILE or
            result.get('max_absolute_winrate_error_exclusive') != 0.075 or
            result.get('teacher', {}).get('quantization') != 'none' or
            result.get('teacher', {}).get('inference_dtype') != 'bfloat16'):
        raise ValueError('Require at least 32 actually mined, consolidated and isolated training labels')
    for section in ['inputs', 'outputs']:
        for path, expected in proof[section].items():
            bound.bind(path, expected)
    config = proof['config']
    has_review = 'prose_review_exclusions' in config
    if (has_review != any(name in result for name in SUMMARY_FIELDS) or
            has_review and (not isinstance(config['prose_review_exclusions'], str) or
                            not config['prose_review_exclusions'].strip())):
        raise ValueError('Prose review configuration and collection evidence differ')
    required = config.get('require_immediate_mate_consistency', False)
    if (type(required) is not bool or
            result.get('immediate_mate_score_consistency_required', False) is not required or
            (required and result.get('immediate_mate_consistency_profile') != IMMEDIATE_MATE_CONSISTENCY_PROFILE) or
            (not required and result.get('immediate_mate_consistency_profile') is not None)):
        raise ValueError('Immediate mate consistency policy differs from collection evidence')
    if required:
        for row in iter_jsonl(data / 'train.jsonl'):
            if not immediate_mate_score_consistency(row, parse_explanation(row['answer']))['valid']:
                raise ValueError('An immediate-mate-filtered collection retains an inconsistent PV')
    mining_path = Path(config.get('mining_manifest') or Path(config['queries']).parent / 'manifest.json')
    mining = bound.json(mining_path)
    if is_student!=bool(mining.get('config',{}).get('student_game_inputs')):
        raise ValueError('Collection and original miner disagree about their explicit source type')
    for section in ['inputs', 'outputs']:
        for path, expected in mining[section].items():
            bound.bind(path, expected)
    if (mining.get('kind') != 'search_distillation_mining' or mining.get('status') != 'complete' or
            mining['config']['checkpoint'] != str(source) or
            proof['inputs'].get(str(mining_path)) != bound.artifacts[str(mining_path)] or
            mining['inputs'].get(str(source)) != bound.bind(source) or
            mining['verification'].get('student_game_training_inputs' if is_student else
                                       'unused_recorded_training_inputs') is not True):
        raise ValueError('Actual search labels must come from this selected parent and unused recorded roots')
    if is_student and (not mining['config'].get('student_game_inputs') or
            Path(mining['config']['student_game_inputs']).resolve()!=Path(student_pool).resolve() or
            mining['config'].get('recorded_inputs') or result.get('recorded_mining_isolation_rechecked_during_collection') is True):
        raise ValueError('Actual neural-game labels lost or disguised their original source type')
    validation_data = Path(result['validation_source_unchanged'])
    for split in ['validation', 'test']:
        original, copied = validation_data / f'{split}.jsonl', data / f'{split}.jsonl'
        if (proof['inputs'].get(str(original)) is None or
                proof['outputs'].get(str(copied)) != proof['inputs'][str(original)]):
            raise ValueError('Distillation must preserve the original protected validation and test bytes')
    return proof


def native_label(row):
    validate_context(row)
    value = parse_explanation(row['answer'])
    facts = validate_explanation(row['fen'], value, require_branches=True)
    prose = validate_prose_moves(row['fen'], value, value['explanation'])
    if not facts['valid'] or not prose['valid']:
        raise ValueError('A consolidated label has invalid native facts or explicit prose moves')
    if row['split'] == 'train':
        rechecked = search_evaluation_filter({'target_fields': value, 'record': row})
        if row.get('search_evaluation_filter') != rechecked:
            raise ValueError('A training label changed its original student evaluation or exclusive filter')
    continuation_positions(row, value)
    return row['feature_key']


def _reopen(proof, bound):
    for section in ['inputs', 'outputs']:
        for path, expected in proof[section].items():
            bound.bind(path, expected)


def _required_binding(proof, section, path, bound):
    expected = proof[section].get(str(path))
    if expected is None:
        raise ValueError('Production evidence must bind the exact original artifact')
    return bound.bind(path, expected)


def _teacher_production(config, queries, responses, teacher, bound):
    """Read the actual inference completion, deployment and untouched responses."""
    path = Path(config['responses']).parent / 'manifest.json'
    inference = bound.json(path)
    saved, result = inference.get('config', {}), inference.get('verification', {})
    if (inference.get('kind') != 'local_teacher_inference' or inference.get('status') != 'complete' or
            saved.get('input') != config['queries'] or saved.get('config') != config['config'] or
            result.get('neural_teacher_inference_executed') is not True or
            type(result.get('examples')) is not int or result['examples'] != len(queries)):
        raise ValueError('Require the completed teacher inference for the exact mined queries')
    _reopen(inference, bound)
    for name in [config['queries'], config['config']]:
        _required_binding(inference, 'inputs', name, bound)
    _required_binding(inference, 'outputs', config['responses'], bound)
    deployment = Path(config['responses']).parent / 'deployment.json'
    identity = bound.json(deployment, _required_binding(inference, 'outputs', deployment, bound))
    if (identity != result.get('teacher') or any(r.get('teacher') != identity for r in responses) or
            identity.get('backend', 'transformers') != saved.get('backend', 'transformers') or
            any(identity.get(k) != teacher[k] for k in
                ['repository', 'revision', 'quantization', 'inference_dtype']) or
            any(not isinstance(teacher.get(k), str) or not teacher[k] for k in ['repository', 'revision']) or
            teacher.get('quantization') != 'none' or teacher.get('inference_dtype') != 'bfloat16'):
        raise ValueError('Teacher responses or deployment differ from the pinned full BF16 teacher')
    weights = Path(teacher['model_path']) / 'weights.manifest.json'
    binding = _required_binding(inference, 'inputs', weights, bound)
    weights_proof = bound.json(weights)
    if (identity.get('weight_manifest_sha256') != binding['sha256'] or
            any(weights_proof.get(k) != teacher[k] for k in ['repository', 'revision', 'quantization']) or
            weights_proof.get('all_official_lfs_sha256_matched') is not True):
        raise ValueError('Teacher deployment lost its official pinned weight inventory')
    observed = identity.get('parameter_elements_by_dtype', {})
    if (not observed.get('torch.bfloat16') or set(observed) - {'torch.bfloat16', 'torch.float32'} or
            any(type(n) is not int or n <= 0 for n in observed.values())):
        raise ValueError('Teacher deployment must contain observed full BF16 parameters')
    if saved.get('backend', 'transformers') == 'vllm':
        workers = identity.get('loaded_worker_identities', [identity.get('loaded_worker_identity')])
        if len(workers) != teacher.get('tensor_parallel_size', 1):
            raise ValueError('Full BF16 teacher worker coverage differs')
        total = Counter()
        for worker in workers:
            dtypes = worker.get('parameter_elements_by_dtype', {}) if isinstance(worker, dict) else {}
            if (not dtypes.get('torch.bfloat16') or set(dtypes) - {'torch.bfloat16', 'torch.float32'} or
                    any(type(n) is not int or n <= 0 for n in dtypes.values()) or
                    worker.get('model_quant_config_is_none') is not True or
                    any(not name.startswith('Unquantized') for name in worker.get('quant_methods', []))):
                raise ValueError('An actual teacher worker uses incompatible or quantized parameters')
            total.update(dtypes)
        if dict(total) != observed:
            raise ValueError('Observed teacher parameter totals differ from its actual workers')
    return {k: identity.get(k) for k in ['repository', 'revision', 'quantization', 'inference_dtype',
        'weight_manifest_sha256', 'generation_profile']}, str(path)


def _production_leaf(data, source, bound, *, source_profile=None):
    # Small valid leaves may contribute; the combined unique-label floor stays 32.
    proof = _leaf_dataset(data, source, bound, minimum_labels=1,source_profile=source_profile)
    config, result = proof['config'], proof['verification']
    for name in [config['queries'], config['responses'], config['config']]:
        _required_binding(proof, 'inputs', name, bound)
    for name in ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'rejected.json']:
        _required_binding(proof, 'outputs', data / name, bound)
    queries, responses = list(iter_jsonl(config['queries'])), list(iter_jsonl(config['responses']))
    teacher = bound.json(config['config'])['search_consolidator']
    if result['teacher'] != teacher or config.get('max_eval_error_exclusive', .075) != .075:
        raise ValueError('Collection teacher or exclusive error filter changed')
    identity, teacher_path = _teacher_production(config, queries, responses, teacher, bound)
    mining_path = Path(config.get('mining_manifest') or Path(config['queries']).parent / 'manifest.json')
    mining = bound.json(mining_path)
    requested_policy = mining['config'].get('acceptance_policy')
    policy = checked_acceptance_evidence(mining['verification'],
        'strict' if requested_policy is None else requested_policy)
    clean_ply = mining['config'].get('min_clean_ply')
    clean_ply = 1 if clean_ply is None else clean_ply
    checked_acceptance_policy(policy, mining['config'].get('root_candidate_policy') or 'strict',
        mining['config'].get('child_contract', 'full'), mining['config'].get('max_root_retries'), clean_ply)
    ledgers = [p for p in mining['outputs'] if Path(p).name == 'results.partial.jsonl']
    if len(ledgers) != 1:
        raise ValueError('Mining must preserve one complete original root-attempt ledger')
    attempts = list(iter_jsonl(ledgers[0]))
    attempted_ids = [a['source_root_id'] for a in attempts]
    if (len(set(attempted_ids)) != len(attempts) or mining['verification'].get('processed') != len(attempts) or
            [a['query'] for a in attempts if a.get('query') is not None] != queries or
            any(a.get('query') is not None and (a.get('reason') != 'accepted_for_consolidation' or
                a['query'].get('source_root_id') != a['source_root_id']) for a in attempts)):
        raise ValueError('Mined queries differ from the preserved original root attempts')
    additional, protection_info = mining_reservations(config['queries'], queries, config['heldout_data'],
        config['validation_data'], bound, mining_path, config.get('recorded_inputs'),
        expected_attempted_root_ids=attempted_ids,student_game_inputs=config.get('student_game_inputs'))
    if result.get('search_protection') != protection_info.get('search_protection'):
        raise ValueError('Collection protection differs from its actual source mining')
    forbidden = reserved_positions(config['heldout_data']) | reserved_positions(config['validation_data']) | additional
    rebuilt, rejected = rebuild_labels(queries, responses, teacher, forbidden,
        require_immediate_mate_consistency=config.get('require_immediate_mate_consistency', False))
    if 'prose_review_exclusions' in config:
        path = config['prose_review_exclusions']
        _required_binding(proof, 'inputs', path, bound)
        rebuilt, rejected, review_summary = exclude_reviewed_prose(rebuilt, rejected, bound.json(path),
            queries_sha256=bound.artifacts[str(config['queries'])]['sha256'],
            responses_sha256=bound.artifacts[str(config['responses'])]['sha256'])
        if any(type(result.get(name)) is not type(review_summary[name]) or
               result[name] != review_summary[name] for name in SUMMARY_FIELDS):
            raise ValueError('Prose exclusion summary differs from its bound case review')
    actual = list(iter_jsonl(data / 'train.jsonl'))
    if (actual != rebuilt or bound.json(data / 'rejected.json') != rejected or
            len(actual) != result['accepted_training_roots'] or
            len(rejected) != result.get('rejected_consolidations') or result.get('mined_queries') != len(queries)):
        raise ValueError('Accepted or rejected labels differ from raw teacher reconstruction')
    lines = []
    for number, raw in enumerate((data / 'train.jsonl').read_bytes().splitlines(keepends=True), 1):
        if raw.strip():
            if not raw.endswith(b'\n'):
                raise ValueError('Byte-preserved JSONL records require original newline terminators')
            lines.append((number, raw))
    if len(lines) != len(actual):
        raise ValueError('Raw training record coverage differs')
    for row in actual:
        native_label(row)
    is_student = bool(mining['config'].get('student_game_inputs'))
    pool = Path(mining['config']['student_game_inputs'] if is_student else mining['config']['recorded_inputs'])
    isolation = {'recorded_pool_manifest': bound.artifacts[str(pool / 'manifest.json')],
        'search_acceptance': {'policy': policy, 'min_clean_ply': clean_ply},
        **{f'base_{s}': bound.artifacts[str(Path(config['heldout_data']) / f'{s}.jsonl')]
           for s in ['validation', 'test']}}
    if 'search_protection' in protection_info:
        isolation['search_protection'] = protection_info['search_protection']
    if source_profile==SHARED_SOURCE_PROFILE:
        from .student_game_inputs import shared_isolation_identity
        pool_proof = bound.json(pool/'manifest.json')
        authority = pool_proof['config']['shared_isolation_authority']['authority'] if is_student else pool
        original_count = protection_info.get('additional_reserved_canonical_and_explanation_positions', len(additional))
        common = shared_isolation_identity(authority,config['heldout_data'],config['validation_data'],bound,original_count)
        isolation.pop('recorded_pool_manifest')
        isolation.update(shared_isolation_authority=common,
            require_immediate_mate_consistency=config.get('require_immediate_mate_consistency',False))
    return proof, actual, lines, identity, attempted_ids, teacher_path, isolation


def _collection_paths(collections):
    if (not isinstance(collections, (list, tuple)) or not collections or
            any(not isinstance(p, (str, Path)) or not str(p) for p in collections)):
        raise ValueError('Require an ordered nonempty list of complete leaf collections')
    paths = [Path(p) for p in collections]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Collection directories must be distinct, including path aliases')
    return paths


def _merge_plan(collections, source, bound, *, source_profile=None):
    _checked_source_profile(source_profile)
    paths = _collection_paths(collections)
    rows, lines, provenance, leaves = [], [], [], []
    kept, ids, manifest_hashes, attempted, accepted_roots = {}, {}, set(), set(), set()
    protected, teacher, teacher_identity, validation_source, isolation = None, None, None, None, None
    mined_queries, rejected_queries = 0, 0
    for data in paths:
        leaf = (_production_leaf(data,source,bound) if source_profile is None else
                _production_leaf(data,source,bound,source_profile=source_profile))
        proof, actual, raw_lines, identity, root_ids, inference_path, leaf_isolation = leaf
        proof_hash = bound.artifacts[str(data / 'manifest.json')]['sha256']
        if proof_hash in manifest_hashes:
            raise ValueError('A collection completion is repeated through an alias')
        manifest_hashes.add(proof_hash)
        copies = {s: bound.artifacts[str(data / f'{s}.jsonl')] for s in ['validation', 'test']}
        if protected is None:
            protected, teacher, teacher_identity = copies, proof['verification']['teacher'], identity
            validation_source = proof['verification']['validation_source_unchanged']
            isolation = leaf_isolation
        elif copies != protected or identity != teacher_identity or isolation != leaf_isolation:
            raise ValueError('Collections must share exact protected splits, recorded pool, acceptance policy and teacher')
        attempted.update(root_ids)
        mined_queries += proof['verification']['mined_queries']
        rejected_queries += proof['verification']['rejected_consolidations']
        leaves.append({'data': str(data), 'manifest_sha256': proof_hash,
                       'accepted_training_records': len(actual), 'attempted_starting_roots': len(root_ids),
                       'teacher_inference_manifest': inference_path})
        if source_profile==SHARED_SOURCE_PROFILE:
            leaves[-1]['source_kind'] = ('student_engine_training_game' if
                proof['verification'].get('student_game_mining_isolation_rechecked_during_collection') is True
                else 'original_recorded_game_or_tactical_source')
        if proof['config'].get('require_immediate_mate_consistency', False):
            leaves[-1]['require_immediate_mate_consistency'] = True
        for row, (number, raw) in zip(actual, raw_lines, strict=True):
            key, identity_id = row['feature_key'], row['id']
            if identity_id in ids and ids[identity_id] != key:
                raise ValueError('One training ID refers to different full history contexts')
            ids[identity_id] = key
            accepted_roots.add(row['search_source_root'])
            duplicate = key in kept
            index = kept[key] if duplicate else len(rows)
            if duplicate and any(row[k] != rows[index][k] for k in ['initial_fen', 'moves', 'history']):
                raise ValueError('A history key refers to different complete histories')
            reference = rows[index] if duplicate else row
            value, original = parse_explanation(row['answer']), parse_explanation(reference['answer'])
            provenance.append({'source_data': str(data), 'source_manifest_sha256': proof_hash,
                'source_line': number, 'raw_row_sha256': hashlib.sha256(raw).hexdigest(),
                'id': identity_id, 'feature_key': key, 'source_root_id': row['search_source_root'],
                'decision': 'duplicate' if duplicate else 'kept', 'kept_record_index': index,
                'answer_differs_from_kept': row['answer'] != reference['answer'],
                'selected_move_differs_from_kept': value['move'] != original['move'],
                'pv_differs_from_kept': value['pv'] != original['pv']})
            if not duplicate:
                kept[key] = index; rows.append(row); lines.append(raw)
    if len(rows) < 32:
        raise ValueError('Require at least 32 unique full-history training labels after deduplication')
    all_rows = list(rows)
    all_ids, all_keys = {r['id'] for r in rows}, set(kept)
    for split in ['validation', 'test']:
        for row in iter_jsonl(paths[0] / f'{split}.jsonl'):
            if (row.get('split') != split or row.get('stage') != 'explanation' or
                    row['id'] in all_ids or row['feature_key'] in all_keys):
                raise ValueError('Protected labels changed split, stage or unique history identity')
            native_label(row)
            all_ids.add(row['id']); all_keys.add(row['feature_key']); all_rows.append(row)
    native_rows = []
    for row in all_rows:
        value = parse_explanation(row['answer'])
        native_rows.append(dict(row, future_moves=list(value['pv']),
            future_branches=[list(branch['pv']) for branch in value['branches']]))
    split_proof = verify_splits(native_rows)
    duplicates = [p for p in provenance if p['decision'] == 'duplicate']
    summary = {**split_proof, 'accepted_training_roots': len(rows), 'raw_accepted_training_records': len(provenance),
        'unique_full_history_contexts': len(rows), 'duplicate_history_records': len(duplicates),
        'duplicate_records_with_different_answers': sum(p['answer_differs_from_kept'] for p in duplicates),
        'duplicate_records_with_different_selected_moves': sum(p['selected_move_differs_from_kept'] for p in duplicates),
        'duplicate_records_with_different_pvs': sum(p['pv_differs_from_kept'] for p in duplicates),
        'unique_attempted_starting_roots': len(attempted), 'unique_accepted_starting_roots': len(accepted_roots),
        'mined_queries': mined_queries, 'rejected_consolidations': rejected_queries,
        'teacher': teacher, 'teacher_identity': teacher_identity, 'collections': leaves,
        'search_acceptance_profile': isolation['search_acceptance'],
        'duplicate_policy': 'first_collection_then_original_line_wins',
        'original_training_record_bytes_preserved': True,
        'all_leaf_accepted_and_rejected_labels_reconstructed': True,
        'all_actual_teacher_deployments_and_responses_reopened': True,
        'neural_consolidation_executed': True, 'structured_facts_verified': True,
        'explicit_prose_moves_natively_verified': True, 'all_branch_positions_isolated': True,
        'full_history_termination_checked': True, 'recorded_mining_isolation_rechecked_during_collection': True,
        'selected_branch_evaluation_filter_applied': True, 'evaluation_filter_profile': EVALUATION_FILTER_PROFILE,
        'max_absolute_winrate_error_exclusive': .075, 'validation_source_unchanged': validation_source,
        'all_strategic_prose_semantics_verified': False, 'strategic_prose_human_rating': False,
        'student_search_or_oracle_comparisons_recomputed': False,
        'new_model_teacher_or_engine_queries': 0, 'student_SGD_executed': False,
        'independent_test_used_for_training_or_selection': False}
    mate_sources = sum(leaf.get('require_immediate_mate_consistency', False) for leaf in leaves)
    if mate_sources:
        summary.update(immediate_mate_consistency_source_collections=mate_sources,
            all_source_collections_require_immediate_mate_consistency=mate_sources == len(leaves),
            immediate_mate_consistency_profile=IMMEDIATE_MATE_CONSISTENCY_PROFILE)
    if source_profile==SHARED_SOURCE_PROFILE:
        summary.update(source_profile=source_profile,
            shared_isolation_authority=isolation['shared_isolation_authority'],
            all_source_mining_isolation_rechecked_during_collection=True,
            source_kind_counts=dict(Counter(leaf['source_kind'] for leaf in leaves)))
        if any(leaf['source_kind']=='student_engine_training_game' for leaf in leaves):
            summary.pop('recorded_mining_isolation_rechecked_during_collection')
    if 'search_protection' in isolation:
        summary['search_protection'] = isolation['search_protection']
    bound.unchanged()
    return paths, lines, provenance, summary


def merge_collections(collections, source, output, *, source_profile=None):
    """Requalify flat leaves, keep first original history bytes, bind every source.

    Runs CPU/native checks only. Call inside the project's verified-input session
    when reopening a production miner with large immutable model/cache inputs.
    Completed and failed outputs are never overwritten.
    """
    output, source = Path(output), Path(source)
    if output.exists():
        raise FileExistsError('Preserve completed and interrupted merges; use a fresh output')
    collections = _collection_paths(collections)
    if any(p.resolve() in output.resolve().parents for p in collections):
        raise ValueError('Merged output must be separate from all immutable leaf directories')
    bound = BoundInputs()
    paths, lines, provenance, summary = _merge_plan(collections, source, bound,source_profile=source_profile)
    output.mkdir(parents=True)
    train = output / 'train.jsonl'
    with train.open('xb') as handle:
        for raw in lines:
            handle.write(raw)
    outputs = [train, output / 'provenance.jsonl']
    write_jsonl(outputs[-1], provenance)
    for split in ['validation', 'test']:
        path = output / f'{split}.jsonl'
        shutil.copyfile(paths[0] / path.name, path); outputs.append(path)
    saved = manifest(MERGED_KIND, {'collections': [str(p) for p in paths], 'checkpoint': str(source),
        'duplicate_policy': summary['duplicate_policy']}, [], outputs, summary)
    if source_profile is not None:
        saved['config']['source_profile'] = source_profile
    saved['inputs'] = bound.artifacts
    bound.unchanged()
    atomic_json(output / 'manifest.json', saved)
    return summary


def collected_dataset(data, source, bound):
    """Reopen a legacy leaf or fully reconstruct a flat merged collection."""
    data = Path(data)
    proof = bound.json(data / 'manifest.json')
    if proof.get('kind') != MERGED_KIND:
        if 'prose_review_exclusions' in proof.get('config', {}):
            # Keep the original 32-label floor, then reconstruct exclusions from
            # the actual raw production records for direct-leaf training too.
            _leaf_dataset(data, source, bound)
            return _production_leaf(data, source, bound)[0]
        return _leaf_dataset(data, source, bound)
    config = proof.get('config', {})
    if proof.get('status') != 'complete' or config.get('checkpoint') != str(source):
        raise ValueError('Merged collection must bind this exact selected search parent')
    actual = BoundInputs()
    paths, lines, provenance, summary = _merge_plan(config.get('collections'), source, actual,
        source_profile=config.get('source_profile'))
    if (proof.get('verification') != summary or config.get('duplicate_policy') != summary['duplicate_policy'] or
            set(proof.get('outputs', {})) != {str(data / name) for name in
                ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'provenance.jsonl']}):
        raise ValueError('Merged completion differs from original leaf reconstruction')
    for path, binding in actual.artifacts.items():
        if proof['inputs'].get(path) != binding:
            raise ValueError('Merged completion lost an original production input binding')
    _reopen(proof, bound)
    if ((data / 'train.jsonl').read_bytes() != b''.join(lines) or
            list(iter_jsonl(data / 'provenance.jsonl')) != provenance or
            any(bound.artifacts[str(data / f'{s}.jsonl')] != actual.artifacts[str(paths[0] / f'{s}.jsonl')]
                for s in ['validation', 'test'])):
        raise ValueError('Merged raw records, duplicate provenance or protected splits changed')
    actual.unchanged(); bound.unchanged()
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collections', nargs='+', required=True, help='Ordered original leaf data directories')
    parser.add_argument('--checkpoint', required=True, help='Exact same selected parent used by every miner')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-profile',choices=[SHARED_SOURCE_PROFILE],
                        help='Explicitly permit typed sources sharing the exact canonical held-out authority')
    args = parser.parse_args()
    from .verified_inputs import input_session_from_environment
    with input_session_from_environment():
        options = {} if args.source_profile is None else {'source_profile':args.source_profile}
        print(json.dumps(merge_collections(args.collections,args.checkpoint,args.output,
            **options)),flush=True)


if __name__ == '__main__':
    main()
