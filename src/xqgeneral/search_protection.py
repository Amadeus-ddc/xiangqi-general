"""Reconstruct the complete held-out footprint for searches after auxiliary SFT."""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from .evidence import atomic_json, code_identity, digest, manifest
from .recorded_search_inputs import BoundInputs
from .verified_inputs import input_session_from_environment

KIND = 'completed_auxiliary_corpus_search_protection'
PROFILE = 'original_authority_plus_all_validation_reference_PVs_and_mirrors_v1'
FILES = ('protected-positions.json', 'counts.json', 'contract.json')

# The earlier producer rejected an unscored best move with RuntimeError.
# The scoring recovery only gave that same rejected case a typed exception.
# Pin the complete previous parser AST, not an open-ended version allowance.
_LEGACY_ORACLE = '2bed40736e2d0e4fe05405783a60bb85073ee92876f79aadf6863ea1c53edb19'
_LEGACY_ORACLE_AST = '41a5f7988759890ffe083890001bf8c167aeca24fd91b9a4b6d58171e5a630e2'
_UNSCORED_CLASS_AST = '1236650141bd26a6a4a5d3bbfbfb4845dd8ac14adbb99cf8906ae42f8e00b769'


def _ast_digest(node):
    canonical = copy.deepcopy(node)
    # Python3.12 added empty type_params fields to ordinary functions/classes.
    # Omit only that empty field; actual generic parameters remain significant.
    for item in ast.walk(canonical):
        if 'type_params' in item._fields and getattr(item, 'type_params', None) == []:
            item._fields = tuple(field for field in item._fields if field != 'type_params')
    return hashlib.sha256(ast.dump(canonical, include_attributes=False).encode()).hexdigest()


def _typed_unscored_upgrade(source):
    """Prove the sole parser edit without importing or running historical code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'UnscoredBestMove']
    if len(classes) != 1 or _ast_digest(classes[0]) != _UNSCORED_CLASS_AST:
        return False
    tree.body.remove(classes[0])
    upgraded = ast.parse('raise UnscoredBestMove(fen, output, nodes)').body[0]
    legacy = ast.parse("raise RuntimeError('Engine completed without a scored best-move candidate')").body[0]

    class RestoreRejectedCase(ast.NodeTransformer):
        replacements = 0

        def visit_Raise(self, node):
            if ast.dump(node) == ast.dump(upgraded):
                self.replacements += 1
                return legacy
            return self.generic_visit(node)

    restore = RestoreRejectedCase();tree = restore.visit(tree)
    return restore.replacements == 1 and _ast_digest(tree) == _LEGACY_ORACLE_AST


def _decoder_compatibility(producer, current):
    if producer == current:
        return None
    if (not isinstance(producer, dict) or set(producer) != set(current) or
            producer.get('oracle.py') != _LEGACY_ORACLE or
            any(producer[name] != value for name, value in current.items() if name != 'oracle.py') or
            not _typed_unscored_upgrade(Path(__file__).with_name('oracle.py').read_text())):
        raise ValueError('Corpus counts or native validation decoding changed')
    return dict(profile='exact_successful_decoder_typed_unscored_error_upgrade_v2',
        producer_oracle_sha256=producer['oracle.py'], current_oracle_sha256=current['oracle.py'],
        previous_full_module_AST_sha256=_LEGACY_ORACLE_AST,
        AST_canonical_profile='omit_only_empty_type_params_v1',
        all_other_closure_modules_byte_identical=True, artifact_source_code_executed=False,
        unscored_best_moves_remain_rejected=True)


def _plan(data, bound):
    # Import lazily: the corpus uses the miner's numerical-question contract.
    from . import oracle_policy_corpus as corpus
    from .oracle_policy_data import _binding
    from .oracle_policy_reference import KIND as REFERENCE_KIND, PROFILE as REFERENCE_PROFILE

    data = Path(data).resolve()
    proof = bound.json(data / 'manifest.json')
    counts = proof.get('verification', {})
    if (proof.get('status') != 'complete' or proof.get('kind') != corpus.KIND or
            proof.get('config', {}).get('profile') != corpus.PROFILE or
            any(counts.get(key) is not True for key in [
                'original_validation_row_bytes_preserved', 'all_supervised_positions_and_mirrors_isolated',
                'no_new_policy_validation_target_enters_other_protected_positions']) or
            counts.get('test_rows') != 0 or counts.get('final_independent_test_rows_parsed_or_tokenized') != 0):
        raise ValueError('Search protection needs the completed isolated seven-stage corpus')
    actual_counts = bound.json(data / 'counts.json', _binding(proof['outputs'], data / 'counts.json'))
    saved = bound.json(data / 'contract.json', _binding(proof['outputs'], data / 'contract.json'))
    if actual_counts != counts:
        raise ValueError('Corpus counts or native validation decoding changed')
    compatibility = _decoder_compatibility(saved.get('closure'), corpus._closure())
    if compatibility is not None and proof.get('code', {}).get('source_sha256', {}).get('src/xqgeneral/oracle.py') != _LEGACY_ORACLE:
        raise ValueError('Historical corpus decoder differs from its bound producer identity')
    shared = saved['original_shared_isolation_authority']
    if shared.get('profile') != 'shared_canonical_heldout_authority_v1':
        raise ValueError('Require the original complete canonical held-out authority')
    authority = Path(shared['authority']).resolve()
    bound.bind(authority / 'manifest.json', shared['authority_manifest_binding'])
    original = bound.json(authority / 'heldout-positions.json', shared['reserved_positions_binding'])
    if (not isinstance(original, list) or any(not isinstance(key, str) for key in original) or
            len(original) != len(set(original)) or len(original) != shared['reserved_positions'] or
            len(original) != counts.get('shared_protected_positions')):
        raise ValueError('Original canonical protected-position coverage changed')
    reserved = set(original)
    primary = Path(saved['explanations'][0]).resolve()
    validation = primary / 'validation.jsonl'
    bound.bind(validation, _binding(proof['inputs'], validation))
    original_labels = list(corpus._lines(validation))
    labels = {row['id']: row for _, _, row in original_labels}
    if len(labels) != len(original_labels):
        raise ValueError('Original explanation validation IDs are not unique')
    path = data / 'validation.jsonl'
    bound.bind(path, _binding(proof['outputs'], path))
    rows = list(corpus._lines(path))
    by_stage = dict(Counter(row['stage'] for _, _, row in rows))
    if (set(by_stage) != set(corpus.ALL_STAGES) or by_stage != counts.get('validation_rows_by_stage') or
            len(rows) != counts.get('validation_question_rows') or
            len({row['id'] for _, _, row in rows}) != len(rows)):
        raise ValueError('Actual seven-stage validation coverage changed')
    for _, _, row in rows:
        corpus._basic(row, 'validation', corpus.ALL_STAGES)
    originals = [(raw, row) for _, raw, row in rows if row['stage'] not in {corpus.NUMERIC_STAGE, 'move_quality'}]
    source_originals, seen = [], {}
    for directory in [*saved['explanations'], *saved['replay'], saved['policies']]:
        source = Path(directory).resolve() / 'validation.jsonl'
        bound.bind(source, _binding(proof['inputs'], source))
        for _, raw, row in corpus._lines(source):
            corpus._basic(row, 'validation', corpus.ALL_STAGES)
            if row['id'] in seen:
                previous_raw, previous_source = seen[row['id']]
                if previous_source == source or previous_raw != raw:
                    raise ValueError('Original validation sources contain conflicting duplicate IDs')
            else:
                seen[row['id']] = (raw, source);source_originals.append((raw, row))
    if [raw for raw, _ in source_originals] != [raw for raw, _ in originals]:
        raise ValueError('Corpus changed original source validation bytes or order')
    explanation_bytes = [raw for raw, row in originals if row['stage'] == 'explanation']
    if (explanation_bytes != [raw for _, raw, _ in original_labels] or
            dict(Counter(row['stage'] for _, row in originals)) != counts.get('original_validation_rows_by_stage')):
        raise ValueError('Corpus lost original explanation bytes, order or course validation rows')
    known = set()
    for _, row in originals:
        known.update(corpus._expanded(corpus._positions(row)))
    if not known <= reserved:
        raise ValueError('The original authority does not cover all original validation targets')
    references = Path(saved['references']).resolve()
    reference = bound.json(references / 'manifest.json', _binding(proof['inputs'], references / 'manifest.json'))
    logical = reference.get('config', {}).get('logical', {})
    if (reference.get('status') != 'complete' or reference.get('kind') != REFERENCE_KIND or
            logical.get('profile') != REFERENCE_PROFILE or Path(logical.get('data', '')).resolve() != primary or
            logical.get('nodes', 0) < 1000000 or reference.get('verification', {}).get('validation_only') is not True):
        raise ValueError('Require the exact original-context validation reference')
    source_identity = _binding(reference['inputs'], validation)
    if source_identity != shared['heldout_label_identities']['validation']:
        raise ValueError('Validation reference belongs to a different explanation source')
    for name in ['roots.jsonl']:
        path = references / name
        identity = _binding(reference['outputs'], path)
        bound.bind(path, identity)
        if identity != _binding(proof['inputs'], path):
            raise ValueError('Corpus and reference disagree about actual raw candidate variations')
    additions, rejected, extra = corpus._reference_views(references, reference, labels, known, reserved)
    expected = b''.join(raw for raw, _ in originals) + b''.join(
        (corpus._json(row) + '\n').encode() for row in additions)
    if expected != (data / 'validation.jsonl').read_bytes() or len(rejected) != counts['rejected_new_validation_questions']:
        raise ValueError('Native reference reconstruction differs from the actual corpus validation')
    reserved.update(extra)
    if len(reserved) != counts.get('extended_protected_positions'):
        raise ValueError('Native complete reference footprint differs from corpus isolation')
    result = dict(profile=PROFILE, shared_protected_positions=len(original),
        all_reference_root_and_candidate_PV_positions=len(extra),
        newly_reserved_positions=len(reserved) - len(original), protected_positions=len(reserved),
        original_validation_rows=len(originals), all_seven_stage_validation_rows=len(rows),
        validation_reference_roots=len(labels), original_explanation_bytes_and_order_preserved=True,
        all_reference_candidates_including_unused_long_suffixes_reserved=True,
        color_mirrors_reserved=True, original_authority_coverage_reused=True,
        independent_final_test_labels_read=False, new_engine_or_model_queries=0, new_SGD_updates=0)
    contract = dict(profile=PROFILE, corpus=str(data), corpus_manifest_binding=bound.artifacts[str(data / 'manifest.json')],
        shared_isolation_authority=shared, native_closure=corpus._closure())
    if compatibility is not None:
        contract['native_decoder_compatibility'] = compatibility
    bound.unchanged()
    return contract, result, reserved


def _completed(path, proof, bound, contract, result, positions, inputs):
    current = code_identity()['source_sha256']['src/xqgeneral/search_protection.py']
    if (proof.get('status') != 'complete' or proof.get('kind') != KIND or proof.get('config') != contract or
            proof.get('inputs') != inputs or proof.get('verification') != result or
            proof.get('code', {}).get('source_sha256', {}).get('src/xqgeneral/search_protection.py') != current or
            set(proof.get('outputs', {})) != {str(path.parent / name) for name in FILES}):
        raise ValueError('Completed search protection differs from native reconstruction')
    for name, binding in inputs.items():
        bound.bind(name, binding)
    for name in FILES:
        value = bound.json(path.parent / name, proof['outputs'][str(path.parent / name)])
        expected = sorted(positions) if name == FILES[0] else result if name == FILES[1] else contract
        if value != expected:
            raise ValueError('Saved protection output differs from complete raw reference coverage')
    bound.unchanged()


def prepare(data, output, *, resume=False):
    bound = BoundInputs()
    contract, result, positions = _plan(data, bound)
    inputs = dict(bound.artifacts)
    root = Path(output).resolve()
    if root.exists():
        if not resume:
            raise FileExistsError('Preserve existing search protection; use a fresh output')
        path = root / 'manifest.json'
        _completed(path, bound.json(path), bound, contract, result, positions, inputs)
        return result
    root.mkdir(parents=True)
    for name, value in zip(FILES, [sorted(positions), result, contract]):
        atomic_json(root / name, value)
    proof = manifest(KIND, contract, [], [root / name for name in FILES], result)
    proof['inputs'] = inputs
    bound.unchanged();atomic_json(root / 'manifest.json', proof)
    return result


def checked_protection(path, bound, *, base_data=None, source_pool=None, corpus=None):
    path = Path(path).resolve()
    proof = bound.json(path)
    declared = proof.get('config', {}).get('corpus')
    if not isinstance(declared, str) or corpus is not None and Path(declared).resolve() != Path(corpus).resolve():
        raise ValueError('Protection must belong to the declared complete training corpus')
    actual = BoundInputs()
    contract, result, positions = _plan(declared, actual)
    _completed(path, proof, bound, contract, result, positions, actual.artifacts)
    shared = contract['shared_isolation_authority']
    if base_data is not None:
        directory = Path(base_data).resolve()
        if set(shared['base_data_identities']) != {str(directory / (s + '.jsonl')) for s in ['train', 'validation', 'test']}:
            raise ValueError('Search base data belongs to a different canonical authority')
    if source_pool is not None:
        pool = Path(source_pool).resolve();source = bound.json(pool / 'manifest.json')
        if source.get('status') != 'complete':
            raise ValueError('Only a completed source pool can use this protection')
        if source.get('kind') == 'prepared_unused_recorded_search_roots':
            if (pool != Path(shared['authority']).resolve() or
                    bound.artifacts[str(pool / 'manifest.json')] != shared['authority_manifest_binding']):
                raise ValueError('Recorded pool differs from the original protected authority')
        elif source.get('kind') == 'prepared_student_training_game_roots':
            if source.get('config', {}).get('shared_isolation_authority') != shared:
                raise ValueError('Student games belong to a different protected authority')
        else:
            raise ValueError('Protection needs an explicit completed recorded or student-game pool')
    actual.unchanged();bound.unchanged()
    return positions, dict(profile=PROFILE, manifest=str(path), manifest_binding=bound.artifacts[str(path)],
        corpus=contract['corpus'], corpus_manifest_binding=contract['corpus_manifest_binding'],
        protected_positions=result['protected_positions'], newly_reserved_positions=result['newly_reserved_positions'],
        all_raw_reference_variations_and_mirrors_reconstructed=True, independent_final_test_labels_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True);parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true');args = parser.parse_args()
    with input_session_from_environment():
        print(json.dumps(prepare(args.data, args.output, resume=args.resume)), flush=True)


if __name__ == '__main__':main()
