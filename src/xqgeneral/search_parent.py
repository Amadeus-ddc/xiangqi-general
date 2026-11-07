"""CPU-only admission of evaluated descendants to the existing search pilot."""
import ast
from pathlib import Path

from .auxiliary_policy_preflight import KIND as AUXILIARY_TOKEN_KIND
from .clean_sft_evaluation import checked_raw, validation_spec
from .distillation_preflight import AUXILIARY_ROUND_KIND, ROUND_KIND, TOKEN_KIND, completed_parent
from .evidence import code_identity, digest, load_jsonl
from .feature_store import feature_paths
from .foundation_handoff import Artifacts
from .review_explanations import DIMENSIONS, blinded_query, parse_rating


INITIAL_TOKEN_KIND = 'latent_only_reviewed_explanation_data_preflight'
INITIAL_ROUND_KIND = 'clean_curriculum_to_initial_sft_pipeline'


def family(token):
    kinds = {INITIAL_TOKEN_KIND: 'clean', AUXILIARY_TOKEN_KIND: 'auxiliary', TOKEN_KIND: 'distilled'}
    if token.get('kind') not in kinds:
        raise ValueError('Search requires a declared initial, auxiliary or search-distillation token stage')
    return kinds[token['kind']]


def round_counts(proof):
    result = proof['verification']
    counts = {key: result.get(key, 0) for key in ['trained_auxiliary_policy_rounds', 'trained_search_distillation_rounds']}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError('Completed auxiliary and search round counts must be nonnegative integers')
    if proof.get('kind') == INITIAL_ROUND_KIND and any(counts.values()):
        raise ValueError('Initial SFT cannot claim completed descendant training rounds')
    return counts


def auxiliary_ancestors(source, proof, token, bound):
    """Follow bound native completions; each actual SGD round increments its own count."""
    corpora, seen = set(), set()
    while proof.get('kind') in {AUXILIARY_ROUND_KIND, ROUND_KIND}:
        key = str(Path(source).resolve())
        if key in seen:
            raise ValueError('Completed search-parent lineage contains a cycle')
        seen.add(key)
        auxiliary = proof['kind'] == AUXILIARY_ROUND_KIND
        if token.get('kind') != (AUXILIARY_TOKEN_KIND if auxiliary else TOKEN_KIND):
            raise ValueError('Search-parent lineage changed its native data preflight stage')
        contract = proof['config']; token_path = contract['token_preflight']
        bound.bind(token_path, proof['inputs'].get(token_path))
        if (token_path not in proof['inputs'] or digest(token_path) != contract['token_preflight_sha256'] or
                bound.json(token_path) != token):
            raise ValueError('Each lineage completion must bind its exact actual token preflight')
        logical = token['config']['logical'] if auxiliary else token['config']
        previous_source = logical['source']; previous_path = logical['parent_completion']
        if (contract.get('source') != previous_source or previous_path not in proof['inputs'] or
                previous_source not in proof['inputs']):
            raise ValueError('Completed lineage must retain its actual selected parent and completion')
        bound.bind(previous_path, proof['inputs'][previous_path])
        _, previous = completed_parent(previous_source, previous_path, bound)
        if previous['outputs'].get(previous_source) != proof['inputs'][previous_source]:
            raise ValueError('Lineage source differs from its native completed selected parent')
        expected = round_counts(previous)
        expected['trained_auxiliary_policy_rounds' if auxiliary else 'trained_search_distillation_rounds'] += 1
        if round_counts(proof) != expected:
            raise ValueError('Lineage must increment only the actual auxiliary or search SGD round count')
        if auxiliary:
            corpora.add(str(Path(logical['data']).resolve()))
        source, proof = previous_source, previous
        if proof.get('kind') in {AUXILIARY_ROUND_KIND, ROUND_KIND}:
            path = proof['config']['token_preflight']
            if path not in proof['inputs']:
                raise ValueError('Ancestor completion lost its actual data preflight binding')
            token = bound.json(path, proof['inputs'][path])
    if proof.get('kind') != INITIAL_ROUND_KIND or any(round_counts(proof).values()):
        raise ValueError('The complete lineage must terminate at the declared clean initial SFT')
    bound.unchanged()
    return corpora


def _function(path, name):
    tree = ast.parse(Path(path).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError('Preserved native replay function is missing or ambiguous')
    return ast.dump(functions[0], include_attributes=False)


def raw_source(config, bound):
    """Read an explicitly pinned source inventory, never execute artifact code."""
    reference = config.get('validation_source_manifest')
    if reference is None:
        return code_identity(), {'historical_raw_source_explicitly_declared': False}
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'} or
            not isinstance(reference['path'], str) or not reference['path'] or
            not isinstance(reference['sha256'], str) or len(reference['sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in reference['sha256'])):
        raise ValueError('Historical validation source needs an explicit manifest path and SHA256')
    path = Path(reference['path'])
    if digest(path) != reference['sha256']:
        raise ValueError('Declared historical validation source manifest changed')
    declared = bound.json(path); inventory = declared.get('source_sha256')
    source = path.parent / 'source/xqgeneral'
    if (set(declared) != {'revision', 'source_sha256'} or not isinstance(declared['revision'], str) or
            not declared['revision'] or not isinstance(inventory, dict) or
            set(inventory) != {p.name for p in source.glob('*.py')} or not inventory or
            any(Path(name).name != name or not name.endswith('.py') for name in inventory)):
        raise ValueError('Historical validation source needs its complete preserved Python inventory')
    for name, sha in inventory.items():
        if bound.bind(source / name)['sha256'] != sha:
            raise ValueError('Declared historical validation source bytes changed')
    current = Path(__file__).resolve().parent
    # Replaying saved records/metrics requires these native consumers to agree.
    # It does not assert that a different inference deployment is equivalent.
    for name in ['evidence.py', 'evaluate_explanations.py', 'foundation_handoff.py', 'feature_store.py',
                 'rules.py', 'move_decoding.py', 'modeling.py', 'expert.py', 'bridge.py', 'board_tokens.py',
                 'review_explanations.py']:
        if inventory.get(name) != digest(current / name):
            raise ValueError('Historical raw replay changed a required native consumer dependency')
    for filename, function in [('clean_sft_evaluation.py', 'checked_raw'), ('sft_preflight.py', 'checked_artifacts'),
                               ('explanations.py', 'parse_explanation')]:
        if _function(source / filename, function) != _function(current / filename, function):
            raise ValueError('Historical raw artifact or metric replay contract changed')
    def question(path):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPLANATION_QUESTION' for t in node.targets):
                return ast.literal_eval(node.value)
        raise ValueError('Preserved explanation prompt constant is missing')
    if question(source / 'explanations.py') != question(current / 'explanations.py'):
        raise ValueError('Historical raw validation used a different explanation prompt')
    return {'revision': declared['revision'], 'source_sha256': {'src/xqgeneral/' + n: sha for n, sha in inventory.items()}}, {
        'historical_raw_source_explicitly_declared': True, 'source_manifest': str(path),
        'source_manifest_binding': bound.artifacts[str(path)], 'replay_checks_saved_records_and_metrics_only': True,
        'different_inference_deployments_claimed_equivalent': False}


def paired_raw(capability_path, capability, source, parent, evaluation, config, bound):
    """Reopen every raw and judged answer, bind its provenance, and recompute metrics."""
    rows, token, _ = validation_spec(evaluation)
    core = Artifacts()
    paths = [str(source), str(Path(evaluation['data']) / 'validation.jsonl'),
             *feature_paths(evaluation['features']), evaluation['executable'], evaluation['weights']]
    for path in dict.fromkeys(map(str, paths)):
        expected = parent['outputs'][path] if path == str(source) else None
        if path == evaluation['features']:
            expected = {key: token['cache_header_only_input'][key] for key in ['sha256', 'bytes']}
        core.add(path, bound.bind(path, expected))
    core.verify()
    expected_code, note = raw_source(config, bound)
    root = Path(capability_path).parent
    if set(capability['verification'].get('raw_validation_by_memory', {})) != {'normal', 'zero', 'shuffled'}:
        raise ValueError('Completed capability must retain all three paired validation memories')
    normal = None
    for memory in evaluation['memories']:
        reference = evaluation.get('reuse_raw_validation', {}).get(memory)
        dest = Path(reference['path']) if reference is not None else root / memory
        path = dest / 'manifest.json'
        if str(path) not in capability['inputs']:
            raise ValueError('Completed capability must bind all three actual raw validation manifests')
        raw = bound.json(path, capability['inputs'][str(path)])
        if reference is not None and digest(path) != reference['manifest_sha256']:
            raise ValueError('Explicit reused raw validation manifest changed')
        for name, binding in raw['outputs'].items():
            if capability['outputs'].get(name) != binding:
                raise ValueError('Capability must retain every original raw, judgment and metric artifact')
            bound.bind(name, binding)
        args = {'checkpoint': str(source), 'data': evaluation['data'], 'features': evaluation['features'],
                'split': 'validation', 'limit': len(rows), 'memory': memory, 'output': str(dest),
                **{key: evaluation[key] for key in ['batch_size', 'max_new_tokens', 'nodes', 'workers', 'seed', 'executable', 'weights']}}
        metrics, judged = checked_raw(dest, memory, rows, args, core, expected_code=expected_code)
        if memory == 'normal': normal = judged
        if capability['verification']['raw_validation_by_memory'][memory] != metrics:
            raise ValueError('Capability header differs from its complete paired raw validation rows')
    _blinded_review(root, capability, normal, evaluation, expected_code, bound)
    bound.unchanged();core.unchanged()
    return dict(note, every_original_paired_raw_answer_and_metric_reopened=True,
                full_blinded_review_raw_responses_and_rejections_reopened=True,
                independent_engine_rejudgment_performed=False, raw_answers_repaired=False)


def _blinded_review(root, capability, normal, evaluation, expected_code, bound):
    def proof(directory, kind):
        path = root / directory / 'manifest.json'
        if str(path) not in capability['inputs']:
            raise ValueError('Completed capability must bind every actual blinded review stage')
        value = bound.json(path, capability['inputs'][str(path)])
        if value.get('status') != 'complete' or value.get('kind') != kind or value.get('code') != expected_code:
            raise ValueError('Blinded review differs from its declared actual producer source')
        for name, binding in value['outputs'].items():
            if capability['outputs'].get(name) != binding:
                raise ValueError('Capability lost an original blinded review output')
            bound.bind(name, binding)
        for name, binding in value['inputs'].items(): bound.bind(name, binding)
        return value
    queries = root / 'queries'; inference = root / 'reviewer'; ratings = root / 'ratings'
    query = proof('queries', 'blinded_neural_explanation_review')
    neural = proof('reviewer', 'local_teacher_inference')
    rating = proof('ratings', 'blinded_neural_explanation_review')
    raw_queries = load_jsonl(queries / 'queries.jsonl')
    expected = {row['id']: blinded_query(row) for row in normal}
    if (query['verification'].get('split') != 'validation' or len(raw_queries) != len(expected) or
            len({row['id'] for row in raw_queries}) != len(raw_queries) or
            any(row != expected.get(row['id']) for row in raw_queries)):
        raise ValueError('Blinded review must cover every unchanged normal answer and its grounded facts')
    teacher = bound.json(evaluation['teacher_config'])['search_consolidator']
    if (neural['config'].get('input') != str(queries / 'queries.jsonl') or
            neural['config'].get('config') != evaluation['teacher_config'] or
            neural['config'].get('max_new_tokens') != evaluation['review_max_new_tokens'] or
            neural['config'].get('backend', 'transformers') != evaluation.get('review_teacher_backend', 'transformers') or
            rating['config'].get('queries') != str(queries / 'queries.jsonl') or
            rating['config'].get('responses') != str(inference / 'responses.jsonl') or
            rating['config'].get('config') != evaluation['teacher_config']):
        raise ValueError('Actual blinded reviewer or rating inputs differ from the evaluated configuration')
    responses = load_jsonl(inference / 'responses.jsonl')
    if len(responses) != len(expected) or len({r['id'] for r in responses}) != len(responses) or {r['id'] for r in responses} != set(expected):
        raise ValueError('Preserve every requested raw reviewer answer, including rejection cases')
    accepted, rejected = {}, set()
    for response in responses:
        try: accepted[response['id']] = dict(id=response['id'], **parse_rating(response, teacher))
        except (ValueError, KeyError, TypeError): rejected.add(response['id'])
    actual = load_jsonl(ratings / 'ratings.jsonl'); bad = bound.json(ratings / 'rejected.json')
    if (len(actual) != len(accepted) or {r['id']: r for r in actual} != accepted or
            len(bad) != len(rejected) or {r['id'] for r in bad} != rejected):
        raise ValueError('Blinded ratings and rejections differ from the unmodified raw teacher responses')
    summary = rating['verification']
    means = {key: sum(r[key] for r in accepted.values()) / len(accepted) if accepted else None for key in DIMENSIONS}
    fraction = sum(all(r[key] >= 4 for key in DIMENSIONS) for r in accepted.values()) / len(expected)
    if (summary.get('requested') != len(expected) or summary.get('valid_ratings') != len(accepted) or
            summary.get('rejected') != len(rejected) or summary.get('mean_scores_conditional_on_valid_ratings') != means or
            summary.get('all_dimensions_at_least_4_fraction_all_queries') != fraction or
            summary.get('ratings_with_unsupported_claims') != sum(bool(r['unsupported_claims']) for r in accepted.values()) or
            capability['verification'].get('blinded_prose_review') != summary):
        raise ValueError('Blinded review aggregates differ from actual accepted and rejected raw ratings')


def admit_descendant(source, parent, token, capability_path, capability, evaluation, config, pool_path, bound):
    auxiliary = family(token) == 'auxiliary'
    expected_kind = AUXILIARY_ROUND_KIND if auxiliary else ROUND_KIND
    if parent.get('kind') != expected_kind:
        raise ValueError('Actual completed search parent differs from its auxiliary or distillation token stage')
    corpora = auxiliary_ancestors(source, parent, token, bound)
    if len(corpora) > 1:
        raise ValueError('Multiple distinct auxiliary corpora require a complete union certificate before search')
    protection = config.get('protection_manifest')
    if corpora and protection is None:
        raise ValueError('Auxiliary descendants require their complete validation search-protection certificate')
    protection_note = None
    if protection is not None:
        from .search_protection import checked_protection
        _, protection_note = checked_protection(protection, bound, base_data=config['data'],
            source_pool=pool_path, corpus=next(iter(corpora)) if corpora else None)
    result = capability['verification']
    flags = {'actual_clean_initial_sft_completed_before_model_or_reviewer_load': False,
             'actual_full_decoder_auxiliary_round_completed_before_evaluation': auxiliary,
             'actual_full_decoder_distillation_round_completed_before_evaluation': not auxiliary}
    if (any(result.get(key) is not value for key, value in flags.items()) or
            any(result.get(key) != value for key, value in round_counts(parent).items())):
        raise ValueError('Capability must preserve the actual native training stage and cumulative round counts')
    note = paired_raw(capability_path, capability, source, parent, evaluation, config, bound)
    return dict(note, completed_ancestor_counts_rechecked=True, auxiliary_corpora=sorted(corpora),
                search_protection=protection_note)
