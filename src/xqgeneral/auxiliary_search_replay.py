"""Deterministic original-task replay with optional original-coaching retention."""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path

from .curriculum_data import STAGES as COURSES
from .evidence import atomic_json, manifest
from .recorded_search_inputs import BoundInputs
from .coaching_replay import coaching_config, original_training_subset

KIND = 'latent_only_auxiliary_search_replay'
PROFILE = 'original_courses_and_oracle_policy_replay_v1'
COACHING_PROFILE = 'original_coaching_courses_and_oracle_policy_replay_v1'
LEGACY_PROFILE = 'original_courses_replay_v1'
STAGES = (*COURSES, 'engine_numeric', 'move_quality')


def native_label(row):
    from .course_tasks import PAPER_PROFILE, grade_answer, validate_context
    from .foundation_preflight import native_answer, validate_task
    # Course questions ask about a future board. Numeric targets instead allow
    # any native terminal outcome, including AXF draws, and their actual PV is
    # checked by the numeric grader below. Single-move tasks have no future target.
    validate_context(row, check_future=row['stage'] in COURSES)
    if row['stage'] in COURSES:
        validate_task(row, PAPER_PROFILE)
        if not grade_answer(native_answer(row), row['answer'], row['task_type'])['correct']:
            raise ValueError('Replay answer differs from its native course rules')
    elif row['stage'] == 'explanation':
        from .reviewed_explanations import checked_label
        from .explanations import parse_explanation
        from .prose_moves import validate_prose_moves
        checked_label(row)
        value = parse_explanation(row['answer'])
        if not validate_prose_moves(row['fen'], value, value['explanation'])['valid']:
            raise ValueError('Original coaching contains invalid explicit prose moves')
    elif row['stage'] in STAGES:
        from .oracle_policy_evaluation import grade
        if not grade(row, row['answer'])['contract_valid']:
            raise ValueError('Replay answer violates its native numeric or move contract')
    else:
        raise ValueError('Replay supports only declared original task stages')
    return row['feature_key']


def selection(data, train_per_stage, seed, bound, *, workers=1,
              coaching_data=None, coaching_train_rows=0):
    from .auxiliary_policy_preflight import corpus_source
    from .oracle_policy_corpus import ALL_STAGES, _basic, _lines
    from .recorded_coach import ordered_native_map
    if type(workers) is not int or workers < 1:
        raise ValueError('Native replay workers must be a positive integer')
    if type(train_per_stage) is not int or train_per_stage < 1 or type(seed) is not int or seed < 0:
        raise ValueError('Use a positive per-stage quota and a nonnegative integer seed')
    coaching = coaching_config(coaching_data, coaching_train_rows)
    data = Path(data).resolve()
    corpus, counts = corpus_source(data, bound)
    queues = {stage: [] for stage in STAGES}
    observed = Counter()
    for pointer, raw, row in _lines(data / 'train.jsonl'):
        _basic(row, 'train', ALL_STAGES)
        observed[row['stage']] += 1
        stage = row['stage']
        if stage not in queues:
            continue
        rank = int.from_bytes(hashlib.sha256(json.dumps([seed, stage, row['id']],
            ensure_ascii=False, separators=(',', ':')).encode()).digest(), 'big')
        item = (-rank, pointer['byte_offset'], raw, row)
        queue = queues[stage]
        if len(queue) < train_per_stage:
            heapq.heappush(queue, item)
        elif rank < -queue[0][0]:
            heapq.heapreplace(queue, item)
    if observed != counts['train']:
        raise ValueError('The original corpus training counts changed')
    train = [(offset, raw, row) for queue in queues.values() for _, offset, raw, row in queue]
    train.sort(key=lambda item: item[0])
    selected = {'train': [(raw, row) for _, raw, row in train], 'validation': []}
    observed = Counter()
    for _, raw, row in _lines(data / 'validation.jsonl'):
        _basic(row, 'validation', ALL_STAGES)
        observed[row['stage']] += 1
        if row['stage'] in STAGES:
            selected['validation'].append((raw, row))
    if observed != counts['validation']:
        raise ValueError('The original corpus validation counts changed')
    if coaching is not None:
        retained, coaching_result = original_training_subset(
            coaching['coaching_data'], coaching['coaching_train_rows'], seed, bound)
        selected['train'].extend(retained)
    ids = set()
    for split, items in selected.items():
        actual = Counter(row['stage'] for _, row in items)
        expected = {s: min(train_per_stage, counts[split][s]) if split == 'train' else counts[split][s]
                    for s in STAGES}
        if coaching is not None and split == 'train':
            expected['explanation'] = coaching_result['selected_training_rows']
        if actual != expected:
            raise ValueError('Every stage must retain its exact selected training and complete validation coverage')
        for _, row in items:
            if row['id'] in ids:
                raise ValueError('Replay rows require unique native history identities')
            ids.add(row['id'])
    rows = [row for items in selected.values() for _, row in items]
    if list(ordered_native_map(native_label, rows, workers)) != [row['feature_key'] for row in rows]:
        raise ValueError('Replay rows require unique native history identities')
    result = dict(by_split={split: len(items) for split, items in selected.items()},
        by_split_and_stage={split: dict(Counter(row['stage'] for _, row in items))
                            for split, items in selected.items()},
        source_rows_by_split_and_stage=counts,
        unique_history_contexts_by_split={split: len({row['feature_key'] for _, row in items})
                                         for split, items in selected.items()},
        all_selected_train_and_validation_answers_natively_regraded=True,
        all_selected_full_histories_natively_checked=True,
        all_selected_supervised_futures_natively_checked=True,
        supervised_future_contract_by_stage={**dict.fromkeys(COURSES, 'paper_course_future_board_v1'),
            'engine_numeric': 'six_ply_or_any_native_terminal_v1', 'move_quality': 'current_position_only_v1'},
        training_rows_drawn_only_from_original_train_split=True,
        original_training_rows_preserved_byte_for_byte=True,
        all_original_six_stage_validation_bytes_and_order_preserved=True,
        deterministic_bounded_training_subset=True, source_corpus_kind=corpus['kind'],
        new_engine_or_teacher_queries=0, independent_test_used_for_training_or_selection=False,
        independent_test_answers_parsed_or_tokenized=0,
        strong_play_or_reliable_coaching_proven=False)
    result['by_split']['test'] = 0
    if coaching is not None:
        result['original_coaching_replay'] = coaching_result
        result['supervised_future_contract_by_stage']['explanation'] = 'native_original_coaching_lines_v1'
    bound.unchanged()
    return selected, result


def checked_subset(directory, bound, *, workers=1, profile=PROFILE):
    if type(workers) is not int or workers < 1:
        raise ValueError('Native replay workers must be a positive integer')
    directory = Path(directory).resolve()
    proof = bound.json(directory / 'manifest.json')
    config = proof.get('config', {})
    if (profile not in (PROFILE, COACHING_PROFILE) or
            proof.get('status') != 'complete' or proof.get('kind') != KIND or
            config.get('profile') != profile or not isinstance(config.get('data'), str)):
        raise ValueError('Require a completed replay subset with the exact explicit profile')
    coaching = {}
    if profile == COACHING_PROFILE:
        coaching = coaching_config(config.get('coaching_data'), config.get('coaching_train_rows'))
        if coaching is None:
            raise ValueError('Coaching retention requires the frozen original source and quota')
    elif 'coaching_data' in config or 'coaching_train_rows' in config:
        raise ValueError('Six-task replay cannot silently include original coaching')
    for name, identity in proof['inputs'].items():
        bound.bind(name, identity)
    for name, identity in proof['outputs'].items():
        bound.bind(name, identity)
    selected, expected = selection(config['data'], config['train_per_stage'], config['seed'], bound,
                                   workers=workers, **coaching)
    for split in ['train', 'validation', 'test']:
        path = directory / (split + '.jsonl')
        bound.bind(path, proof['outputs'].get(str(path)))
        raw = b''.join(item[0] for item in selected.get(split, []))
        if path.read_bytes() != raw:
            raise ValueError('Replay must be the independently reconstructed original byte subsequence')
    if proof.get('verification') != expected:
        raise ValueError('Replay completion differs from reconstructed native coverage')
    bound.unchanged()
    return [row for items in selected.values() for _, row in items], expected


def prepare(data, output, *, train_per_stage=2048, seed=20261107, resume=False,
            coaching_data=None, coaching_train_rows=0, workers=1):
    root, data = Path(output).resolve(), Path(data).resolve()
    coaching = coaching_config(coaching_data, coaching_train_rows)
    profile = COACHING_PROFILE if coaching is not None else PROFILE
    config = dict(profile=profile, data=str(data), train_per_stage=train_per_stage, seed=seed)
    config.update(coaching or {})
    bound = BoundInputs()
    if root.exists():
        if not resume:
            raise FileExistsError('Preserve prior replay attempts; choose a fresh output')
        proof = bound.json(root / 'manifest.json')
        if proof.get('config') != config:
            raise ValueError('Replay resume must keep its frozen source, quota and seed')
        _, result = checked_subset(root, bound, workers=workers, profile=profile)
        return result
    selected, result = selection(data, train_per_stage, seed, bound, workers=workers, **(coaching or {}))
    root.mkdir(parents=True)
    outputs = []
    for split in ['train', 'validation', 'test']:
        path = root / (split + '.jsonl')
        path.write_bytes(b''.join(raw for raw, _ in selected.get(split, [])))
        outputs.append(path)
    bound.unchanged()
    proof = manifest(KIND, config, outputs=outputs, verification=result)
    proof['inputs'] = bound.artifacts
    atomic_json(root / 'manifest.json', proof)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-per-stage', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--coaching-data', help='Explicit qualified original coaching dataset')
    parser.add_argument('--coaching-train-rows', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    from .verified_inputs import input_session_from_environment
    with input_session_from_environment():
        print(json.dumps(prepare(**vars(parser.parse_args()))), flush=True)


if __name__ == '__main__':
    main()
