"""Balanced, raw-generation board QA evaluation; validation guides adaptation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import time
from .evidence import atomic_json, load_jsonl, manifest, write_jsonl
from .curriculum_data import STAGES
from .course_tasks import PAPER_PROFILE, grade_answer, row_profile


def balanced_rows(rows, per_task, seed, stages=None):
    if type(per_task) is not int or per_task < 1:
        raise ValueError('QA examples per task must be a positive integer')
    groups = defaultdict(list)
    for row in rows:
        if stages is not None and row['stage'] not in stages:
            continue
        groups[(row['stage'], row['task_type'])].append(row)
    rng = random.Random(seed)
    selected = []
    for group in sorted(groups):
        selected += rng.sample(groups[group], min(per_task, len(groups[group])))
    rng.shuffle(selected)
    return selected


def normalized(answer):
    return re.sub(r'\s+', ' ', answer.strip())


def qa_summary(records):
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['stage']}/{r['task_type']}"].append(r['correct'])
    summary = {'examples': len(records), 'accuracy': sum(r['correct'] for r in records) / len(records),
               'by_task': {k: {'n': len(v), 'accuracy': sum(v) / len(v)} for k, v in sorted(groups.items())}}
    profiles = {row_profile(r) for r in records}
    if len(profiles) != 1:
        raise ValueError('Raw QA cannot mix task profiles')
    if profiles == {PAPER_PROFILE}:
        summary.update(task_profile=PAPER_PROFILE, prose_graded=False)
        for key in ('answer_format_valid', 'canonical_tag_correct', 'raw_exact_correct'):
            summary[key + '_rate'] = sum(r[key] for r in records) / len(records)
    return summary


def prediction_record(row, answer):
    """Retain raw text and apply only the declared answer-comparison contract."""
    result = {'id': row['id'], 'game_id': row['game_id'], 'stage': row['stage'],
              'task_type': row['task_type'], 'question': row['question'],
              'question_variant': row.get('question_variant', 0),
              'expected': row['answer'], 'generated': answer}
    if row_profile(row) == PAPER_PROFILE:
        result.update(task_profile=PAPER_PROFILE, **grade_answer(row['answer'], answer, row['task_type']))
    else:
        result['correct'] = normalized(answer) == normalized(row['answer'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/research-v1')
    parser.add_argument('--features', default='data/research-v1/features.pt')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--per-task', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--memory', choices=['normal', 'zero', 'shuffled'], default='normal')
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--stages', nargs='+', choices=STAGES)
    parser.add_argument('--question-formats', type=int, choices=[1, 3], default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from .inference import Predictor
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh evaluation output')
    path = Path(args.data) / f'{args.split}.jsonl'
    rows = balanced_rows(load_jsonl(path), args.per_task, args.seed, args.stages)
    if not rows or args.batch_size < 1:
        raise ValueError('QA evaluation requires nonempty selected tasks and a positive batch size')
    profiles = {row_profile(row) for row in rows}
    if len(profiles) != 1:
        raise ValueError('QA evaluation requires one declared task profile')
    profile = next(iter(profiles))
    if args.question_formats == 3:
        from .selfplay_grounding import question_variant
        counts = defaultdict(int)
        varied = []
        for row in rows:
            key = row['stage'], row['task_type']
            variant = counts[key] % 3
            counts[key] += 1
            varied.append(dict(row, question=question_variant(row, variant), question_variant=variant))
        rows = varied
    model = Predictor(args.checkpoint, feature_cache=args.features)
    if profile != row_profile(model.config):
        raise ValueError('Evaluation data and checkpoint task profiles differ')
    if profile == PAPER_PROFILE:
        args.task_profile = profile
    start, results = time.monotonic(), []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        if args.memory == 'shuffled' and len(batch) == 1:
            raise ValueError('Select a sample/batch size without a singleton shuffle batch')
        outputs = model.generate_batch(batch, args.max_new_tokens, args.memory)
        for row, answer in zip(batch, outputs, strict=True):
            results.append(prediction_record(row, answer))
        print(json.dumps({'evaluated': len(results), 'seconds': time.monotonic() - start}), flush=True)
    dest.mkdir(parents=True)
    write_jsonl(dest / 'predictions.jsonl', results)
    proof = {**qa_summary(results), 'raw_generation': True, 'oracle_used': False,
             'split': args.split, 'seconds': time.monotonic() - start}
    atomic_json(dest / 'metrics.json', proof)
    atomic_json(dest / 'manifest.json', manifest('balanced_board_qa', vars(args),
                [args.checkpoint, path, args.features], [dest / 'predictions.jsonl', dest / 'metrics.json'], proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
