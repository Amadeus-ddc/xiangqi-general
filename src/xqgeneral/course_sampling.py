"""Author-style answer frequencies and distinct queries for native courses."""
from collections import Counter, defaultdict

from .course_tasks import DIAGONALS, PAPER_PROFILE, PAPER_TASKS, target_fen, traced_square
from .curriculum_data import SQUARES, SYMBOLS
from .rules import in_check, piece_map

SAMPLING_PROFILE = 'author_answer_frequency_xiangqi_v1'
TRAIN_QUERIES = {
    'piece': 4, 'locate': 2, 'file': 2, 'rank': 2, 'diagonal': 3, 'counts': 1, 'materials': 1,
    'moves': 4, 'controllers': 4, 'captures': 1, 'checks': 1, 'parries': 1, 'mate': 1,
}
SHAPED_TASKS = {'piece', 'locate', 'moves', 'controllers'}
EMPTY = 'piece:empty'
CHANGED, SAME = 'distinct:changed', 'distinct:same'


def piece_class(symbol):
    return 'piece:' + symbol if symbol is not None else EMPTY


def square_class(square):
    return 'square:' + square


def make_sampler(profile):
    if profile is None:
        return None
    if profile != SAMPLING_PROFILE:
        raise ValueError('Unknown paper course sampling profile')
    return FrequencySampler()


class FrequencySampler:
    """Counters belong to one completed corpus, independently per split and task."""
    def __init__(self):
        self.frequencies = defaultdict(Counter)

    def query_groups(self, root, fen, rng, future, *, only_kind=None, only_task=None):
        split = root['split']
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Adaptive questions require a preassigned source split')
        board = piece_map(fen)
        initial = piece_map(root['fen']) if future else None
        suffix = 'future' if future else 'current'
        groups = []
        for kind in ('static', 'dynamic'):
            stage, questions = kind + '_' + suffix, []
            for task in PAPER_TASKS[stage]:
                if only_kind is not None and kind != only_kind or only_task is not None and task != only_task:
                    continue
                if task == 'parries' and not in_check(fen):
                    continue
                frequency = self.frequencies[(split, stage, task)] if task in SHAPED_TASKS else {}
                if task in ('piece', 'controllers'):
                    entities = list(SQUARES)
                elif task == 'locate':
                    entities = list(SYMBOLS)
                elif task == 'moves':
                    entities = sorted(s for s, p in board.items() if p.isupper() == (fen.split()[1] == 'w'))
                elif task == 'file':
                    entities = list('abcdefghi')
                elif task == 'rank':
                    entities = list(range(10))
                elif task == 'diagonal':
                    entities = list(DIAGONALS)
                else:
                    entities = [None]
                count = TRAIN_QUERIES[task] if split == 'train' else 1
                for _ in range(min(count, len(entities))):
                    entity = self._choose(task, entities, board, initial, frequency, rng)
                    entities.remove(entity)
                    query = ({'square': entity} if task in ('piece', 'controllers') else
                             {'symbol': entity} if task == 'locate' else
                             {'source': entity} if task == 'moves' else
                             {'file': entity} if task == 'file' else
                             {'rank': entity} if task == 'rank' else
                             {'start': entity[0], 'end': entity[-1]} if task == 'diagonal' else {})
                    questions.append((task, query))
            groups.append(questions)
        return groups

    @staticmethod
    def _choose(task, entities, board, initial, frequency, rng):
        if task == 'piece':
            if initial is not None:
                changed = [s for s in entities if initial.get(s) != board.get(s)]
                same = [s for s in entities if initial.get(s) == board.get(s)]
                if changed and same:
                    wc, ws = 1 / (frequency.get(CHANGED, 0) + 1), 1 / (frequency.get(SAME, 0) + 1)
                    entities = changed if rng.random() < wc / (wc + ws) else same
                else:
                    entities = changed or same
            classes = [piece_class(board.get(s)) for s in entities]
            multiplicities = Counter(classes)
            weights = [1 / ((frequency.get(p, 0) + 1) * multiplicities[p] *
                            (frequency.get(square_class(s), 0) + 1)) for p, s in zip(classes, entities)]
        elif task == 'locate':
            answers = [tuple(square_class(s) for s in sorted(board) if board[s] == p) or (EMPTY,)
                       for p in entities]
            multiplicities = Counter(answers)
            weights = [1 / ((sum(frequency.get(c, 0) for c in answer) + 1) * multiplicities[answer])
                       for answer in answers]
        elif task in ('moves', 'controllers'):
            if task == 'controllers':
                empty = [s for s in entities if s not in board]
                occupied = [s for s in entities if s in board]
                if (rng.random() < .5 and empty) or not occupied:
                    return rng.choice(empty)
                entities = occupied
            classes = [piece_class(board[s]) for s in entities]
            multiplicities = Counter(classes)
            weights = [1 / ((frequency.get(p, 0) + 1) * multiplicities[p]) for p in classes]
        elif task == 'diagonal':
            weights = [len(d) for d in entities]
        else:
            return rng.choice(entities)
        return rng.choices(entities, weights=weights, k=1)[0]

    def observe(self, records):
        """Advance only after the source root's distinct queries are all rendered."""
        for row in records:
            task = row['task_type']
            if row.get('task_profile') != PAPER_PROFILE or row['split'] not in ('train', 'validation', 'test'):
                raise ValueError('Adaptive frequencies require native paper rows and reserved splits')
            if task not in SHAPED_TASKS:
                continue
            board, query = piece_map(target_fen(row)), row['query']
            if task == 'piece':
                square = query['square']
                classes = [square_class(square), piece_class(board.get(square))]
                if row['future_moves']:
                    changed = piece_map(row['fen']).get(square) != board.get(square)
                    classes.append(CHANGED if changed else SAME)
            elif task == 'locate':
                classes = [square_class(s) for s in sorted(board) if board[s] == query['symbol']] or [EMPTY]
            elif task == 'moves':
                classes = [piece_class(board[traced_square(row)])]
            else:
                classes = [piece_class(board.get(query['square']))]
            self.frequencies[(row['split'], row['stage'], task)].update(classes)

    def verification(self):
        return {'sampling_profile': SAMPLING_PROFILE,
                'global_answer_class_frequency_shaping_applied': True,
                'author_training_query_multiplicity_applied': True,
                'training_queries_requested_per_root_by_task': dict(TRAIN_QUERIES),
                'heldout_queries_requested_per_root_by_task': 1,
                'distinct_queries_capped_to_available_entities': True,
                'frequency_counts_include_written_color_mirrors': True,
                'frequencies_independent_per_split_stage_task': True,
                'answer_class_frequency_by_split_stage_task': {
                    '/'.join(group): dict(sorted(frequency.items()))
                    for group, frequency in sorted(self.frequencies.items())}}
