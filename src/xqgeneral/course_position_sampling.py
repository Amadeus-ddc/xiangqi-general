"""Finite, task-specific native root budgets and locked future source mixtures."""
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
from itertools import groupby
import json
import multiprocessing
import random
import sqlite3

from .course_tasks import PAPER_TASKS, validate_context
from .evidence import history_key, position_key
from .engine_selfplay import context_positions
from .course_source_semantics import native_classes, root_footprint
from .rules import in_check, legal_moves, replay
from .symmetry import mirror_fen

POSITION_PROFILE = 'author_task_position_budget_mix_xiangqi_v1'
SPLITS = ('train', 'validation', 'test')
TRAIN_POSITIONS = {'piece': 457143, 'locate': 914286, 'file': 914286, 'rank': 914286,
                   'diagonal': 609524, 'counts': 1828572, 'materials': 1828572,
                   'moves': 457143, 'controllers': 457143, 'captures': 1828572,
                   'checks': 1828572, 'parries': 1828572, 'mate': 360000}
CLASSES = ('generic', 'have_check', 'in_check', 'mate', 'stalemate', 'near_mate_check')
CLASS_BITS = {label: 1 << i for i, label in enumerate(CLASSES)}
MIXES = {'checks': (('generic', .2), ('have_check', .8)),
         'parries': (('in_check', 1.),),
         'mate': (('mate', .30), ('near_mate_check', .20), ('in_check', .10),
                  ('stalemate', .05), ('generic', .35))}


def position_budgets(value=None):
    if value is None:
        return {split: {stage: {task: TRAIN_POSITIONS[task] if split == 'train' else
                                        100 if split == 'validation' else 1000 for task in tasks}
                       for stage, tasks in PAPER_TASKS.items()} for split in SPLITS}
    if not isinstance(value, dict) or set(value) != set(SPLITS):
        raise ValueError('Position budgets must declare all three preassigned splits')
    for stages in value.values():
        if not isinstance(stages, dict) or set(stages) != set(PAPER_TASKS):
            raise ValueError('Position budgets must declare every paper course')
        for stage, tasks in stages.items():
            if (not isinstance(tasks, dict) or set(tasks) != set(PAPER_TASKS[stage]) or
                    any(type(n) is not int or n < 1 for n in tasks.values())):
                raise ValueError('Position budgets require positive counts for every declared task')
    return deepcopy(value)


def root_targets(root):
    """Recompute classes for every horizon; never trust a saved difficulty tag."""
    validate_context(root, check_future=True)
    for branch in root.get('future_branches', []):
        validate_context(dict(root, future_moves=branch), check_future=True)
    future = root.get('future_moves', [])
    generated = root.get('extension_is_recorded_source_move') is False
    witness = root.get('paper_generated_terminal_witness' if generated else 'paper_recorded_mate_witness', [])
    distance = None
    if witness:
        if (not isinstance(witness, list) or len(witness) > 14 or future != witness[:len(future)] or
                generated and (len(witness) != 1 or root.get('paper_recorded_mate_witness'))):
            raise ValueError('Declared mate witness differs from the locked actual future')
        complete = replay(root['initial_fen'], root['moves'] + witness)
        validate_context(dict(root, moves=root['moves'] + witness, history=complete, fen=complete[-1],
                              feature_key=history_key(complete), future_moves=[]), check_future=True)
        if in_check(complete[-1]) and not legal_moves(complete[-1]):
            distance = len(witness)
        elif not generated:
            raise ValueError('Recorded mate witness does not end in native checkmate')
    elif in_check(root['fen']) and not legal_moves(root['fen']):
        distance = 0
    fens = replay(root['fen'], future)
    targets = [{'horizon': h, 'fen': fen, 'classes': native_classes(fen, None if distance is None else distance - h)}
               for h, fen in enumerate(fens)]
    saved = root.get('paper_source_targets')
    if saved is not None:
        if (not isinstance(saved, list) or len(saved) != len(targets) or
                any(row.get('horizon') != target['horizon'] or row.get('fen') != target['fen'] or
                    row.get('classes') != target['classes'] for row, target in zip(saved, targets, strict=True))):
            raise ValueError('Saved native difficulty targets differ from their source history')
        field = 'plies_before_generated_mate' if generated else 'plies_before_recorded_mate'
        if any(row.get(field) != (None if distance is None else distance - row['horizon']) for row in saved):
            raise ValueError('Saved mate distance differs from its declared native witness')
    return targets


def weighted_interleave(streams, seen, counters):
    """Author smooth weighted round robin, one finite pass through each source."""
    active = [{'label': label, 'it': iter(values), 'weight': weight, 'credit': 0.}
              for label, values, weight in streams]
    while active:
        total = sum(s['weight'] for s in active)
        for source in active:
            source['credit'] += source['weight']
        source = max(active, key=lambda s: s['credit'])
        source['credit'] -= total
        for candidate in source['it']:
            if candidate['query_key'] in seen or candidate['color_key'] in seen:
                counters['duplicates'][source['label']] += 1
                continue
            break
        else:
            counters['exhausted'][source['label']] += 1
            active.remove(source)
            continue
        seen.update((candidate['query_key'], candidate['color_key']))
        counters['selected'][source['label']] += 1
        yield source['label'], candidate


def catalog_inputs(roots, workers=1):
    """Bounded ordered native workers; catalog IDs remain in original file order."""
    if type(workers) is not int or workers < 1:
        raise ValueError('Positive native position worker count required')
    if workers == 1:
        for root in roots:
            yield root, root_targets(root)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for root in roots:
            pending.append((root, pool.submit(root_targets, root)))
            if len(pending) >= workers * 2:
                source, job = pending.popleft()
                yield source, job.result()
        for source, job in pending:
            yield source, job.result()


class PositionCatalog:
    """Disk-backed source JSON; iterators hold only root keys and eligible horizons."""
    def __init__(self, path, *, prior_games=None, prior_positions=None):
        self.db = sqlite3.connect(path)
        self.db.executescript('''
            CREATE TABLE roots (id INTEGER PRIMARY KEY, feature TEXT UNIQUE NOT NULL,
                split TEXT NOT NULL, query_key TEXT NOT NULL, color_key TEXT NOT NULL, value TEXT NOT NULL);
            CREATE TABLE targets (root INTEGER NOT NULL, horizon INTEGER NOT NULL, mask INTEGER NOT NULL,
                variant_move TEXT, descendant_feature TEXT);
            CREATE INDEX target_order ON targets(root, horizon);
            CREATE TABLE game_owners (game TEXT PRIMARY KEY, split TEXT NOT NULL);
            CREATE TABLE positions (position TEXT PRIMARY KEY, split TEXT NOT NULL);
        ''')
        self.prior_games = prior_games or {}
        self.db.executemany('INSERT INTO positions VALUES (?, ?)', (prior_positions or {}).items())
        self.counts = Counter()
        self.generated_descendants = []

    def close(self):
        self.db.close()

    def add(self, root, targets=None):
        targets = root_targets(root) if targets is None else targets
        split, game = root['split'], root['game_id']
        if split not in SPLITS or not isinstance(game, str) or not game:
            raise ValueError('Source root has no valid preassigned game split')
        owner = self.db.execute('SELECT split FROM game_owners WHERE game=?', (game,)).fetchone()
        if (owner is not None and owner[0] != split or
                game in self.prior_games and self.prior_games[game] != split):
            raise ValueError('Source root game differs from its preassigned prior corpus split')
        self.db.execute('INSERT OR IGNORE INTO game_owners VALUES (?, ?)', (game, split))
        # Old general roots have no mate-witness fields.
        footprint_root = dict(root, paper_recorded_mate_witness=root.get('paper_recorded_mate_witness', []))
        footprint = root_footprint(footprint_root) | context_positions(root)
        footprint.update(position_key(mirror_fen(key)) for key in list(footprint))
        for key in footprint:
            owner = self.db.execute('SELECT split FROM positions WHERE position=?', (key,)).fetchone()
            if owner is not None and owner[0] != split:
                raise ValueError('Source root, maximum future, witness or color crosses corpus splits')
            self.db.execute('INSERT OR IGNORE INTO positions VALUES (?, ?)', (key, split))
        try:
            row = self.db.execute('INSERT INTO roots(feature,split,query_key,color_key,value) VALUES (?,?,?,?,?)',
                (root['feature_key'], split, position_key(root['fen']), position_key(mirror_fen(root['fen'])),
                 json.dumps(root, ensure_ascii=False)))
        except sqlite3.IntegrityError as error:
            raise ValueError('Source pools contain a duplicate full-history root') from error
        self.db.executemany('INSERT INTO targets VALUES (?,?,?,?,?)',
            ((row.lastrowid, t['horizon'], sum(CLASS_BITS[c] for c in t['classes']), None, None) for t in targets))
        if (root.get('extension_is_recorded_source_move') is False and root['ply'] == 1 and
                root['moves'] == [root['generated_terminal_move']]):
            self.generated_descendants.append((history_key(root['history'][:-1]), root))
        self.counts[split] += 1

    def finish(self):
        # The terminal producer keeps one repeated parent but all distinct legal
        # descendants. Recover every bound one-ply alternative for future tasks.
        for feature, child in self.generated_descendants:
            row = self.db.execute('SELECT id,value FROM roots WHERE feature=?', (feature,)).fetchone()
            if row is None:
                continue
            root_id, value = row
            parent = json.loads(value)
            move = child['generated_terminal_move']
            if parent['game_id'] != child['game_id']:
                continue
            if (parent.get('extension_is_recorded_source_move') is not False or
                    parent['split'] != child['split'] or parent['history'] != child['history'][:-1]):
                raise ValueError('Generated terminal descendant differs from its supplied parent')
            if parent.get('future_moves') == [move]:
                continue
            mate = in_check(child['fen']) and not legal_moves(child['fen'])
            for horizon, fen, distance in [(0, parent['fen'], 1 if mate else None), (1, child['fen'], 0 if mate else None)]:
                mask = sum(CLASS_BITS[c] for c in native_classes(fen, distance))
                self.db.execute('INSERT INTO targets VALUES (?,?,?,?,?)',
                    (root_id, horizon, mask, move, child['feature_key']))
        self.db.execute('CREATE INDEX split_order ON roots(split,id)')
        self.db.commit()

    def candidates(self, split, stage, label, seed):
        current = not stage.endswith('_future')
        rows = self.db.execute('''SELECT r.id,r.feature,r.query_key,r.color_key,t.horizon,t.variant_move,t.descendant_feature
            FROM roots r JOIN targets t ON t.root=r.id
            WHERE r.split=? AND (t.horizon=0 AND ? OR t.horizon>0 AND NOT ?) AND (t.mask & ?) != 0
            ORDER BY r.id,t.horizon,t.variant_move''', (split, current, current, CLASS_BITS[label]))
        for root_id, group in groupby(rows, key=lambda row: row[0]):
            values = list(group)
            _, feature, key, color, *_ = values[0]
            rng = random.Random(int(hashlib.sha256(f'{seed}/{stage}/{label}/{feature}'.encode()).hexdigest(), 16))
            selected = rng.choice(values)
            yield {'root_id': root_id, 'query_key': key, 'color_key': color, 'horizon': selected[4],
                   'variant_move': selected[5], 'descendant_feature': selected[6]}

    def selected_root(self, candidate, stage, task, label):
        value = self.db.execute('SELECT value FROM roots WHERE id=?', (candidate['root_id'],)).fetchone()[0]
        root = json.loads(value)
        root['position_complete_source_future_moves'] = root.get('future_moves', [])
        if candidate['variant_move']:
            root['future_moves'] = [candidate['variant_move']]
            root['paper_generated_terminal_witness'] = [candidate['variant_move']]
            root['generated_terminal_move'] = candidate['variant_move']
            root['position_selected_extension_descendant_feature'] = candidate['descendant_feature']
            root['provenance'] += ';bound_generated_terminal_descendant_course_variant'
        root['future_moves'] = root.get('future_moves', [])[:candidate['horizon']]
        root['position_sampling_profile'] = POSITION_PROFILE
        root['position_source_class'] = label
        root['position_selected_horizon'] = candidate['horizon']
        root['position_selected_stage'] = stage
        root['position_selected_task'] = task
        # This metadata describes the complete pool, not the shortened query.
        if 'paper_source_targets' in root:
            root['paper_complete_source_targets'] = root.pop('paper_source_targets')
        return root

    def selections(self, budgets, seed, report):
        for split in ('test', 'validation', 'train'):
            for stage, tasks in PAPER_TASKS.items():
                seen = set()
                for task in tasks:
                    requested = budgets[split][stage][task]
                    mix = MIXES.get(task, (('generic', 1.),))
                    counters = {kind: Counter() for kind in ('selected', 'duplicates', 'exhausted')}
                    streams = [(label, self.candidates(split, stage, label, seed), weight) for label, weight in mix]
                    selected = 0
                    plain = []
                    for label, candidate in weighted_interleave(streams, seen, counters):
                        selected += 1
                        if task in MIXES:
                            yield self.selected_root(candidate, stage, task, label), stage, task
                        else:
                            plain.append((label, candidate))
                        if selected == requested:
                            break
                    if task not in MIXES:
                        rng = random.Random(int(hashlib.sha256(f'{seed}/{split}/{stage}/{task}/position-order'.encode()).hexdigest(), 16))
                        rng.shuffle(plain)
                        for label, candidate in plain:
                            yield self.selected_root(candidate, stage, task, label), stage, task
                    report[f'{split}/{stage}/{task}'] = {
                        'requested_positions': requested, 'selected_positions': selected,
                        'shortfall_positions': requested - selected,
                        'configured_source_fractions': dict(mix),
                        'selected_positions_by_source': dict(counters['selected']),
                        'duplicate_query_or_color_positions_skipped_by_source': dict(counters['duplicates']),
                        'exhausted_sources': dict(counters['exhausted'])}


def position_verification(budgets, report):
    return {'position_sampling_profile': POSITION_PROFILE,
            'requested_positions_by_split_stage_task': budgets,
            'task_position_selection': report,
            'per_task_root_budgets_applied': True, 'paper_difficult_source_mix_applied': True,
            'future_horizon_locked_before_difficulty_selection': True,
            'bound_generated_terminal_descendant_variants_available_to_future_tasks': True,
            'query_board_and_color_disjoint_across_tasks_within_each_course_split': True,
            'source_streams_are_finite_and_never_cycled': True,
            'single_source_root_selection_shuffled_before_answer_frequency_rendering': True,
            'mixed_source_root_selection_keeps_weighted_interleave_order': True,
            'exhausted_sources_dropped_and_remaining_weights_renormalized': True,
            'actual_selection_shortfalls_reported': True,
            'all_requested_task_root_budgets_filled': all(not r['shortfall_positions'] for r in report.values()),
            'independent_test_used_for_training_or_selection': False}
