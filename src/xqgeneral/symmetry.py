"""Color/rank symmetry, preserving full histories and split ownership."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from .curriculum_data import verify_splits
from .evidence import atomic_json, history_key, load_jsonl, manifest, position_key, write_jsonl
from .rules import legal_moves, play, replay

NAMES = [('红车', '黑车'), ('红马', '黑马'), ('红炮', '黑炮'), ('红仕', '黑士'),
         ('红相', '黑象'), ('红兵', '黑卒'), ('红帅', '黑将'), ('红方', '黑方')]
NAME_SWAP = {a: b for a, b in NAMES} | {b: a for a, b in NAMES}
NAME_PATTERN = re.compile('|'.join(NAME_SWAP))
SQUARE_PATTERN = re.compile(r'[a-i][0-9]')


def mirror_square(square):
    return square[0] + str(9 - int(square[1]))


def mirror_move(move):
    return mirror_square(move[:2]) + mirror_square(move[2:])


def mirror_fen(fen):
    parts = fen.split()
    parts[0] = '/'.join(rank.swapcase() for rank in reversed(parts[0].split('/')))
    parts[1] = 'b' if parts[1] == 'w' else 'w'
    return ' '.join(parts)


def mirror_text(text):
    text = NAME_PATTERN.sub(lambda m: NAME_SWAP[m.group()], text)
    text = SQUARE_PATTERN.sub(lambda m: mirror_square(m.group()), text)
    text = re.sub(r'第([0-9])行', lambda m: f'第{9-int(m.group(1))}行', text)
    return re.sub(r'(?<![A-Za-z0-9_])rank([0-9])(?![0-9])',
                  lambda m: f'rank{9-int(m.group(1))}', text)


def mirrored_qa(row, context=None):
    if context is None:
        moves = [mirror_move(m) for m in row['moves']]
        initial = mirror_fen(row['initial_fen'])
        history = replay(initial, moves)
    else:
        initial, moves, history = context
    from .course_tasks import PAPER_PROFILE, mirror_query, paper_answer, paper_question, row_profile
    if row_profile(row) == PAPER_PROFILE:
        result = dict(row, id=row['id'] + '-color-mirror', fen=history[-1], initial_fen=initial,
                      moves=moves, history=history, feature_key=history_key(history),
                      query=mirror_query(row['query']),
                      future_moves=[mirror_move(m) for m in row.get('future_moves', [])],
                      provenance=row['provenance'] + ';color_rank_symmetry', augmentation_parent=row['id'])
        if 'future_branches' in row:
            result['future_branches'] = [[mirror_move(m) for m in line] for line in row['future_branches']]
        result['question'] = paper_question(result, result.get('question_variant', 0))
        result['answer'] = paper_answer(result)
        return result
    query = dict(row['query'])
    for key in ['square', 'source']:
        if key in query:
            query[key] = mirror_square(query[key])
    if 'move' in query:
        query['move'] = mirror_move(query['move'])
    if 'symbol' in query:
        query['symbol'] = query['symbol'].swapcase()
    if 'rank' in query:
        query['rank'] = 9 - query['rank']
    if 'red' in query:
        query['red'] = not query['red']
    answer = mirror_text(row['answer'])
    if row['task_type'] in {'locate', 'rank', 'moves', 'captures', 'checks'} and answer != '无':
        answer = ' '.join(sorted(answer.split()))
    branches = ({'future_branches': [[mirror_move(m) for m in line]
                                    for line in row['future_branches']]}
                if 'future_branches' in row else {})
    return dict(row, **branches, id=row['id'] + '-color-mirror', fen=history[-1], initial_fen=initial,
                moves=moves, history=history, feature_key=history_key(history), query=query,
                question=mirror_text(row['question']), answer=answer,
                future_moves=[mirror_move(m) for m in row.get('future_moves', [])],
                provenance=row['provenance'] + ';color_rank_symmetry', augmentation_parent=row['id'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/research-v1')
    parser.add_argument('--output', default='data/research-balanced-v1')
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh augmentation output')
    originals = [r for s in ['train', 'validation', 'test']
                 for r in load_jsonl(Path(args.data) / f'{s}.jsonl')]
    ownership = {}
    for row in originals:
        fen = row['fen']
        ownership[position_key(fen)] = row['split']
        for m in row.get('future_moves', []):
            fen = play(fen, m)
            ownership[position_key(fen)] = row['split']
    grouped = {}
    for row in originals:
        grouped.setdefault(row['feature_key'], []).append(row)
    rows, rejected = list(originals), 0
    for records in grouped.values():
        original = records[0]
        initial = mirror_fen(original['initial_fen'])
        moves = [mirror_move(m) for m in original['moves']]
        context = initial, moves, replay(initial, moves)
        converted = [mirrored_qa(r, context) for r in records]
        positions = set()
        for row in converted:
            fen = row['fen']; positions.add(position_key(fen))
            for m in row.get('future_moves', []):
                fen = play(fen, m); positions.add(position_key(fen))
        split = records[0]['split']
        if any(p in ownership and ownership[p] != split for p in positions):
            rejected += 1
            continue
        ownership.update({p: split for p in positions})
        rows += converted
    proof = verify_splits(rows)
    roots = {r['feature_key']: r['fen'].split()[1] for r in rows}
    summary = {**proof, 'records': len(rows), 'side_to_move': dict(Counter(roots.values())),
               'augmentation': 'color_rank_symmetry; parent game stays in the same split',
               'rejected_mirror_roots_for_leakage': rejected}
    outputs = []
    for split in ['train', 'validation', 'test']:
        path = dest / f'{split}.jsonl'
        write_jsonl(path, [r for r in rows if r['split'] == split]); outputs.append(path)
    atomic_json(dest / 'manifest.json', manifest('balanced_curriculum_augmentation', vars(args),
                [Path(args.data) / f'{s}.jsonl' for s in ['train', 'validation', 'test']], outputs, summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
