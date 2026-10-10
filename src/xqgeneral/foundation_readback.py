"""Read back completed foundation preparation against its original game sources."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from .evidence import atomic_json, digest, history_key, manifest, position_key
from .course_tasks import PAPER_PROFILE, recipe_profile
from .foundation_preflight import clean_recipe, corpus_preserves_prefix
from .human_games import assigned_split
from .symmetry import mirror_fen, mirror_move


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def checked_artifacts(proof, outputs=True):
    if proof['status'] != 'complete':
        raise ValueError('Readback requires completed preparation manifests')
    artifacts = dict(proof['inputs'])
    if outputs:
        artifacts.update(proof['outputs'])
    for name, item in artifacts.items():
        if Path(name).stat().st_size != item['bytes'] or digest(name) != item['sha256']:
            raise ValueError(f'Changed completed preparation artifact: {name}')
        print(json.dumps({'completed_artifact_rehashed': name}), flush=True)
    return artifacts


def checked_question_contexts(contexts, cached_keys, expected_count):
    keys, recorded = set(), {}
    for row in contexts:
        key = history_key(row['history'])
        if (key != row['feature_key'] or key not in cached_keys or key in keys or
                len(row['history']) != len(row['moves']) + 1 or
                row['history'][0] != row['initial_fen'] or row['history'][-1] != row['fen']):
            raise ValueError('Question context differs from its cached complete-history identity')
        keys.add(key)
        if row.get('recorded_source_kind'):
            recorded[(row['initial_fen'], tuple(row['moves']))] = row
    if len(keys) != expected_count:
        raise ValueError('Question history coverage differs from the completed native preflight')
    return keys, recorded


def checked_recorded_roots(roots, source_games, contexts, cached_keys, prior, split_seed):
    """Check original source continuations and both colors of every maximum future."""
    heldouts = {'validation': set(), 'test': set()}
    prior_positions = prior['train'] | prior['heldout']
    training, counts, games, kinds = set(), Counter(), set(), Counter()
    seen = set()
    for row in roots:
        source = source_games[row['game_id']]
        ply, future_count = row['ply'], len(row['future_moves'])
        if (row['split'] != source['split'] or
                row['split'] != assigned_split(row['game_id'], split_seed) or
                row['initial_fen'] != source['initial_fen'] or
                row['moves'] != source['moves'][:ply] or row['history'] != source['history'][:ply + 1] or
                row['future_moves'] != source['moves'][ply:ply + future_count] or
                not 1 <= future_count <= 8 or row['recorded_source'] != source['source'] or
                row['recorded_source_kind'] != source['source_kind']):
            raise ValueError('Recorded root differs from its original source game, split or continuation')
        original = contexts[(row['initial_fen'], tuple(row['moves']))]
        mirror = contexts[(mirror_fen(row['initial_fen']), tuple(mirror_move(m) for m in row['moves']))]
        if (original['history'] != row['history'] or original['fen'] != row['fen'] or
                original['feature_key'] != row['feature_key'] or row['feature_key'] in seen or
                [f.split()[:5] for f in mirror['history']] !=
                [mirror_fen(f).split()[:5] for f in row['history']] or
                history_key(mirror['history']) != mirror['feature_key'] or
                mirror['feature_key'] not in cached_keys):
            raise ValueError('Recorded root or native color counterpart has an inconsistent history key')
        seen.add(row['feature_key'])
        future = source['history'][ply:ply + future_count + 1]
        footprint = {position_key(fen) for fen in future}
        footprint.update(position_key(mirror_fen(fen)) for fen in future)
        if row['split'] == 'train':
            training.update(footprint)
        else:
            if footprint.intersection(prior_positions):
                raise ValueError('New recorded holdout overlaps a prior training or holdout future')
            heldouts[row['split']].update(footprint)
        counts[row['split']] += 1
        games.add(row['game_id'])
        kinds[row['recorded_source_kind']] += 1
        if sum(counts.values()) % 8192 == 0:
            print(json.dumps({'recorded_source_prefixes_read_back': sum(counts.values())}), flush=True)
    if heldouts['validation'].intersection(heldouts['test']):
        raise ValueError('Recorded validation and test have overlapping maximum future footprints')
    if training.intersection(prior['heldout'] | heldouts['validation'] | heldouts['test']):
        raise ValueError('Recorded training overlaps an old or new heldout maximum future')
    return {'roots_by_split': dict(counts), 'games_contributing_roots': len(games),
            'roots_by_source_kind': dict(kinds), 'all_original_source_prefixes_and_futures_match': True,
            'native_color_history_keys_match_question_contexts': True,
            'new_holdout_overlap_prior_training_and_holdouts': 0,
            'new_train_overlap_old_and_new_holdouts': 0}


def checked_preserved_cache(cache, previous):
    count = len(previous['keys'])
    if (cache['keys'][:count] != previous['keys'] or cache['depths'] != previous['depths'] or
            len(cache['features']) != len(previous['features'])):
        raise ValueError('Extended cache changed the previous key order or expert layers')
    for index, (new, old) in enumerate(zip(cache['features'], previous['features'], strict=True)):
        if new.dtype != old.dtype or not torch.equal(new[:count], old):
            raise ValueError(f'Extended cache changed previous feature values at layer {index}')
        print(json.dumps({'previous_feature_layer_bitwise_preserved': index}), flush=True)
    if cache['wdl'].dtype != previous['wdl'].dtype or not torch.equal(cache['wdl'][:count], previous['wdl']):
        raise ValueError('Extended cache changed previous WDL values')
    return count


def checked_course_sources(data, data_proof, contexts, cached_keys, expected_contexts, output):
    """Route source reconstruction through the declared data producer contract."""
    data = Path(data)
    question_keys, recorded = checked_question_contexts(contexts, cached_keys, expected_contexts)
    profile = data_proof['config'].get('task_profile', 'legacy')
    corpus_preserves_prefix(data_proof, profile)
    if profile == PAPER_PROFILE:
        from .paper_curriculum import readback
        result = readback(data, output)
        source_proof_path = Path(output) / 'manifest.json'
        result = {'all_supplied_paper_roots_and_question_bytes_regenerated': True,
                  'paper_questions_regenerated': result['questions_regenerated_and_compared'],
                  'paper_source_counts': result['counts'], 'task_profile': PAPER_PROFILE}
        return question_keys, result, [source_proof_path]
    source_games, source_inputs = {}, []
    for folder in data_proof['config']['games']:
        source, source_proof_path = Path(folder) / 'games.jsonl', Path(folder) / 'manifest.json'
        source_proof = json.loads(source_proof_path.read_text())
        if source_proof['status'] != 'complete' or digest(source) != source_proof['outputs'][str(source)]['sha256']:
            raise ValueError('Original recorded game source differs from its completed import')
        source_inputs.extend([source, source_proof_path])
        for game in rows(source):
            source_games.setdefault(game['game_id'], game)
    footprint_root = Path(data_proof['config']['footprints'])
    footprint_path, footprint_proof_path = footprint_root / 'positions.json', footprint_root / 'manifest.json'
    footprint_proof = json.loads(footprint_proof_path.read_text())
    if (footprint_proof['status'] != 'complete' or
            digest(footprint_path) != footprint_proof['outputs'][str(footprint_path)]['sha256']):
        raise ValueError('Reserved prior complete future footprints changed')
    prior = {key: set(value) for key, value in json.loads(footprint_path.read_text()).items()}
    roots_path = data / 'recorded-roots.jsonl'
    if digest(roots_path) != data_proof['outputs'][str(roots_path)]['sha256']:
        raise ValueError('Recorded course roots changed after production generation')
    result = checked_recorded_roots(rows(roots_path), source_games, recorded, cached_keys,
                                   prior, data_proof['config']['split_seed'])
    if (result['roots_by_split'] != data_proof['verification']['original_recorded_roots_by_split'] or
            sum(result['roots_by_split'].values()) != data_proof['verification']['original_recorded_roots']):
        raise ValueError('Recorded root coverage differs from the completed production corpus')
    return question_keys, result, [roots_path, footprint_path, footprint_proof_path, *source_inputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--preflight', required=True, help='Completed foundation_preflight output directory')
    parser.add_argument('--base-proof', required=True, help='Pinned pretrained base-asset identity manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--public-evidence', required=True)
    parser.add_argument('--cpu-threads', type=int, default=8)
    args = parser.parse_args()
    root, recipe_path = Path(args.output), Path(args.config)
    if root.exists() or args.cpu_threads < 1:
        raise ValueError('Use a fresh readback output and positive CPU thread budget')
    recipe = json.loads(recipe_path.read_text())
    clean_recipe(recipe)
    profile = recipe_profile(recipe)
    proof_path = Path(args.preflight) / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    if Path(proof['config']['config']).resolve() != recipe_path.resolve():
        raise ValueError('Completed preflight used a different training recipe')
    root.mkdir(parents=True)
    checked_artifacts(proof)
    observed = proof['verification']
    data = Path(recipe['data_path'])
    data_proof_path = data / 'manifest.json'
    data_proof = json.loads(data_proof_path.read_text())
    corpus_preserves_prefix(data_proof, profile)
    cache_proof_path = Path(recipe['feature_path']).with_suffix('.manifest.json')
    cache_proof = json.loads(cache_proof_path.read_text())
    counts = data_proof['verification']['counts']
    if (data_proof['status'] != 'complete' or cache_proof['status'] != 'complete' or
            observed['total_rule_records'] != sum(counts.values()) or
            observed['train_validation_records_tokenized'] != counts['train'] + counts['validation'] or
            observed['extra_validation_format_encodings'] != 2 * counts['validation'] or
            observed['source_recipe_sha256'] != digest(recipe_path) or
            set(observed['maximum_tokens_by_stage']) != set(recipe['raw_qa_gates']['targets']) or
            any(n > recipe['max_tokens'] for n in observed['maximum_tokens_by_stage'].values()) or
            observed['split_verification'] != data_proof['verification']['split_verification'] or
            observed['split_verification']['game_overlap'] or
            observed['split_verification']['root_and_future_overlap'] or
            observed['actual_student_training_started'] is not False):
        raise ValueError('Preflight counts, training contract or complete future isolation differ')
    base_path = Path(args.base_proof)
    base_proof = json.loads(base_path.read_text())
    base = base_proof['verification']
    if (base['model_revision'] != recipe['model_revision'] or
            base['all_base_asset_bytes_match_cached_pinned_download_etags'] is not True or
            base['trained_course_or_sft_weights_used'] is not False or
            any(Path(f['path']).parent.resolve() != Path(recipe['model_path']).resolve() for f in base['files'])):
        raise ValueError('Base identity differs from the untouched pinned pretrained model slot')
    checked_artifacts(base_proof)
    cache = torch.load(recipe['feature_path'], map_location='cpu', weights_only=True, mmap=True)
    cached_keys = set(cache['keys'])
    if (cache['depths'] != recipe['expert_feature_depths'] or len(cached_keys) != len(cache['keys']) or
            len(cache['keys']) != cache_proof['verification']['roots'] or
            len(cache['keys']) != observed['cache_keys']):
        raise ValueError('Cache keys or expert depths differ from the completed preflight')
    contexts_path = Path(args.preflight) / 'unique-contexts.jsonl'
    question_keys, source_result, source_inputs = checked_course_sources(
        data, data_proof, rows(contexts_path), cached_keys,
        observed['actual_unique_histories_natively_replayed'], root / 'question-source-readback')
    old_cache_path = Path(cache_proof['config']['base_cache'])
    old_proof_path = Path(cache_proof['config']['base_proof'])
    old_proof = json.loads(old_proof_path.read_text())
    if (old_proof['status'] != 'complete' or
            digest(old_cache_path) != old_proof['outputs'][str(old_cache_path)]['sha256']):
        raise ValueError('Previous immutable expert cache changed')
    previous = torch.load(old_cache_path, map_location='cpu', weights_only=True, mmap=True)
    torch.set_num_threads(args.cpu_threads)
    preserved_count = checked_preserved_cache(cache, previous)
    if preserved_count != cache_proof['verification']['base_roots']:
        raise ValueError('Previous cache key coverage differs from extension generation')
    result = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
              'total_rule_records': observed['total_rule_records'], 'records_by_split': counts,
              'train_validation_records_tokenized': observed['train_validation_records_tokenized'],
              'extra_validation_format_encodings': observed['extra_validation_format_encodings'],
              'maximum_tokens_by_stage': observed['maximum_tokens_by_stage'],
              'actual_cached_full_histories': len(cached_keys), 'actual_question_full_histories': len(question_keys),
              'actual_previous_cached_histories_bitwise_preserved': preserved_count,
              'all_preflight_input_output_hashes_read_back': True,
              'official_base_assets_match_pinned_cached_download_hashes': True,
              'official_remote_independently_requeried': False,
              'decoder_literal_board_text': None, 'course_or_sft_checkpoint_loaded': False,
              'student_training_performed_by_readback': False, 'student_improvement_measured': False,
              **source_result}
    atomic_json(root / 'verification.json', result)
    inputs = [recipe_path, proof_path, base_path, data_proof_path, cache_proof_path, contexts_path,
              *source_inputs, old_cache_path, old_proof_path]
    atomic_json(root / 'manifest.json', manifest('recorded_foundation_source_and_cache_readback', vars(args),
                                               inputs, [root / 'verification.json'], result))
    atomic_json(args.public_evidence, dict(result, manifest_sha256=digest(root / 'manifest.json')))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
