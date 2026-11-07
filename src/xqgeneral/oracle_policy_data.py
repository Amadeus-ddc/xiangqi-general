"""Replay native engine ledgers into isolated move and numeric training views."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
from tempfile import mkdtemp

from .evidence import atomic_json, code_identity, digest, history_key, manifest, position_key
from .oracle import parse_analysis
from .policy_data import QUESTION as MOVE_QUESTION, move_label
from .recorded_search_inputs import BoundInputs, checked_root_offset
from .rules import adjudicate, legal_moves, play, replay, validate_position
from .sampling import checked_seed
from .search_distillation import NUMERIC_CHILD_QUESTION
from .student_game_inputs import shared_reservations
from .symmetry import mirror_fen
from .verified_inputs import input_session_from_environment

KIND = 'prepared_oracle_policy_views'
PROFILE = 'unrestricted_engine_policy_views_v1'
NUMERIC_STAGE = 'engine_numeric'
FILES = ('train.jsonl', 'validation.jsonl', 'test.jsonl', 'rejected-queries.jsonl',
         'rejected-contexts.jsonl', 'duplicate-queries.jsonl', 'counts.json', 'contract.json')
CLOSURE = ('oracle_policy_data.py', 'oracle.py', 'rules.py', 'policy_data.py', 'symmetry.py',
           'evidence.py', 'student_game_inputs.py', 'recorded_search_inputs.py')
_RESERVED, _MIN_NODES, _FAMILIES = None, None, None


class Excluded(ValueError):
    """An expected, recorded curation exclusion rather than source corruption."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _read(path):
    return json.loads(Path(path).read_text())


def _binding(group, path):
    if not isinstance(group, dict):
        raise ValueError('Source artifact bindings must be a mapping')
    matches = [value for name, value in group.items() if Path(name).resolve() == Path(path).resolve()]
    if len(matches) != 1:
        raise ValueError('A source must bind each required artifact exactly once')
    return matches[0]


def _same_path(value, expected):
    return isinstance(value, (str, Path)) and Path(value).resolve() == Path(expected)


def _closure():
    current = code_identity()['source_sha256']
    return {name: current['src/xqgeneral/'+name] for name in CLOSURE}


def _context_key(row):
    raw = json.dumps([row['initial_fen'], row['moves']], sort_keys=True,
                     ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def checked_query(item, minimum_nodes=100000):
    """Reparse a successful unrestricted query; never infer scores from bounds."""
    if (not isinstance(item, dict) or set(item) != {'record', 'searchmoves', 'requested_nodes', 'response'} or
            not isinstance(item['record'], dict) or not isinstance(item['searchmoves'], list) or
            type(item['requested_nodes']) is not int or item['requested_nodes'] < 1):
        raise ValueError('Native oracle ledger schema differs')
    row = item['record']
    fields = {'id', 'feature_key', 'game_id', 'split', 'initial_fen', 'moves', 'history', 'fen'}
    if (set(row) != fields or row.get('split') != 'train' or
            any(not isinstance(row.get(k), str) or not row[k] for k in fields-{'moves', 'history'}) or
            any(not isinstance(row.get(k), list) or any(not isinstance(v, str) for v in row[k])
                for k in ['moves', 'history'])):
        raise ValueError('Only native training-owned complete-context records are allowed')
    if item['searchmoves']:
        raise Excluded('restricted_query')
    if item['requested_nodes'] < minimum_nodes:
        raise Excluded('below_minimum_requested_nodes')
    validate_position(row['initial_fen'])
    history = replay(row['initial_fen'], row['moves'])
    if history != row['history'] or history[-1] != row['fen']:
        raise ValueError('Oracle record differs from its available full history')
    if adjudicate(row['initial_fen'], row['moves'])['ended']:
        raise Excluded('terminal_history')
    response = item['response']
    if (not isinstance(response, dict) or response.get('requested_nodes') != item['requested_nodes'] or
            not isinstance(response.get('raw_output'), list) or not response['raw_output'] or
            any(not isinstance(line, str) for line in response['raw_output'])):
        raise ValueError('The original raw engine output and requested budget are required')
    try:
        parsed = parse_analysis(row['fen'], response['raw_output'], item['requested_nodes'])
    except (ValueError, RuntimeError, IndexError, TypeError) as error:
        raise ValueError('Successful oracle record cannot be natively reparsed') from error
    if parsed != response:
        raise ValueError('Stored oracle response differs from its native raw-log reparse')
    best = parsed['candidates'][0]
    if (best['move'] != parsed['best_move'] or best['pv'][0] != best['move'] or
            best['score_type'] not in {'cp', 'mate'} or type(best['score']) is not int or
            best['perspective'] != 'side_to_move'):
        raise ValueError('The exact best-move score contract differs')
    return _context_key(row), best


def _isolated_positions(item, reserved):
    row, response = item['record'], item['response']
    if history_key(row['history']) != row['feature_key']:
        raise Excluded('history_feature_key_mismatch')
    choices = response['candidates']
    positions = {row['fen']}
    for choice in choices:
        fen = row['fen']
        for move in choice['pv']:
            fen = play(fen, move)
            positions.add(fen)
    if any(position_key(fen) in reserved for fen in positions):
        raise Excluded('reserved_root_or_recorded_candidate_PV')
    if any(position_key(mirror_fen(fen)) in reserved for fen in positions):
        raise Excluded('reserved_color_mirror_root_or_candidate_PV')
    return len(positions)


def _numeric_after_isolation(item):
    row, response = item['record'], item['response']
    choices = response['candidates']
    moves = [choice['move'] for choice in choices]
    if len(moves) != min(3, len(legal_moves(row['fen']))) or len(set(moves)) != len(moves):
        raise Excluded('missing_exact_root_candidates')
    pv, termination = [], None
    for move in choices[0]['pv'][:6]:
        pv.append(move)
        outcome = adjudicate(row['initial_fen'], [*row['moves'], *pv])
        if outcome['ended']:
            termination = outcome
            break
    if len(pv) < 6 and termination is None:
        raise Excluded('short_nonterminal_best_PV')
    best = choices[0]
    target = dict(move=best['move'], pv=pv, candidates=moves,
                  evaluation=dict(type=best['score_type'], value=best['score'], perspective='side_to_move'))
    return target, termination


def numeric_target(item, reserved):
    """Apply common isolation and the numerical view's own six-ply contract."""
    positions = _isolated_positions(item, reserved)
    target, termination = _numeric_after_isolation(item)
    return target, termination, positions


def _worker_init(reserved, minimum_nodes, families):
    global _RESERVED, _MIN_NODES, _FAMILIES
    _RESERVED, _MIN_NODES, _FAMILIES = reserved, minimum_nodes, families


def _query_family(row, roots):
    if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or
            any(not isinstance(row.get(k), list) for k in ['moves', 'history']) or
            any(not isinstance(move, str) for move in row['moves'])):
        raise ValueError('Oracle query lost its original recorded-root identity')
    root = roots.get(row['id'][:len('recorded-search-')+64])
    if root is None:
        raise ValueError('Oracle query lies outside its declared native mining root range')
    prefix = len(root['moves'])
    extra = row['moves'][prefix:]
    if (row.get('game_id') != root['game_id'] or row.get('initial_fen') != root['initial_fen'] or
            row.get('split') != 'train' or row['moves'][:prefix] != root['moves'] or
            row['history'][:len(root['history'])] != root['history'] or
            len(row['history']) != len(row['moves'])+1 or
            row['id'] != root['id']+''.join('-'+move for move in extra)):
        raise ValueError('Oracle query differs from its declared recorded root and available history')
    return root, len(extra)


def _worker(task, reserved=None, minimum_nodes=None, families=None):
    reserved = _RESERVED if reserved is None else reserved
    minimum_nodes = _MIN_NODES if minimum_nodes is None else minimum_nodes
    families = _FAMILIES if families is None else families
    destination = Path(task['work'])
    destination.mkdir()
    counts, actual = Counter(), hashlib.sha256()
    with Path(task['ledger']).open('rb') as source, (destination/'decisions.jsonl').open('w') as decisions, (destination/'excluded.jsonl').open('w') as excluded:
        source.seek(task['start'])
        offset = task['start']
        for number in range(task['first_line'], task['first_line']+task['records']):
            raw = source.readline()
            if not raw.endswith(b'\n') or not raw.strip():
                raise ValueError('Oracle ledger contains an incomplete or empty record')
            actual.update(raw)
            counts['raw_queries'] += 1
            query = dict(ledger=task['ledger'], mining_manifest=task['manifest'],
                mining_manifest_sha256=task['manifest_sha256'], line=number, byte_offset=offset,
                raw_line_sha256=hashlib.sha256(raw).hexdigest())
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError('Native oracle query must be an object')
            root, extra = _query_family(item.get('record'), families[task['ledger']])
            query.update(source_root_id=root['id'], source_root_kind=root['recorded_source_kind'],
                         extra_plies_after_recorded_root=extra)
            if 'recorded_source_context' in root:
                query['recorded_source_context'] = root['recorded_source_context']
            if item.get('requested_nodes') != task['requested_nodes']:
                raise ValueError('Query budget differs from the completed mining contract')
            try:
                key, best = checked_query(item, minimum_nodes)
            except Excluded as error:
                reason = str(error)
                counts['excluded_query_'+reason] += 1
                excluded.write(_json(dict(query=query, reason=reason))+'\n')
            else:
                counts['native_checked_potential_records'] += 1
                decision = dict(key=key, signature=[best['move'], best['score_type'], best['score']],
                                query=query, common_reason=None, numeric_reason=None)
                try:
                    checked_positions = _isolated_positions(item, reserved)
                except Excluded as error:
                    decision['common_reason'] = str(error)
                else:
                    decision.update(record=item['record'], checked_positions=checked_positions)
                    try:
                        target, terminal = _numeric_after_isolation(item)
                    except Excluded as error:
                        decision['numeric_reason'] = str(error)
                    else:
                        decision.update(target=target, termination=terminal)
                decisions.write(_json(decision)+'\n')
            offset += len(raw)
        if offset != task['end'] or actual.hexdigest() != task['sha256']:
            raise ValueError('A source chunk changed during native replay')
    return dict(counts)


def _sources(mining, authority, data, executable, weights, seed, bound):
    result = []
    current = code_identity()['source_sha256']
    for directory in mining:
        path = Path(directory)/'manifest.json'
        saved = bound.json(path)
        config, verified = saved.get('config', {}), saved.get('verification', {})
        if (saved.get('kind') != 'search_distillation_mining' or saved.get('status') != 'complete' or
                not _same_path(config.get('output'), directory) or
                not _same_path(config.get('recorded_inputs'), authority) or
                not _same_path(config.get('data'), data) or
                config.get('seed') != seed or
                verified.get('training_roots_only') is not True or
                verified.get('all_successful_oracle_queries_preserved') is not True or
                verified.get('restricted_move_queries_in_oracle_ledger') is not True or
                type(config.get('nodes')) is not int or config['nodes'] < 1 or
                type(config.get('limit')) is not int or config['limit'] < 1 or
                type(verified.get('oracle_query_records')) is not int or verified['oracle_query_records'] < 1 or
                type(verified.get('processed')) is not int or verified['processed'] < 1):
            raise ValueError('Require completed training-only native recorded-source mining ledgers')
        for name in ['oracle.py', 'rules.py']:
            key = 'src/xqgeneral/'+name
            if saved.get('code', {}).get('source_sha256', {}).get(key) != current[key]:
                raise ValueError('The original engine parser or native rules differ')
        for key, supplied in [('executable', executable), ('weights', weights)]:
            if not _same_path(config.get(key), supplied):
                raise ValueError('The explicit engine assets differ from their source contract')
            bound.bind(supplied, _binding(saved.get('inputs'), supplied))
        ledger = Path(directory)/'oracle-queries.partial.jsonl'
        bound.bind(ledger, _binding(saved.get('outputs'), ledger))
        result.append(dict(ledger=str(ledger), manifest=str(path), manifest_sha256=digest(path),
            requested_nodes=config['nodes'], records=verified['oracle_query_records'],
            original_roots=verified['processed'], root_offset=checked_root_offset(config.get('root_offset', 0)),
            root_limit=config['limit']))
    return result


def _root_families(sources, authority, bound):
    """Select the exact ranges after the shared reader validated the entire catalog."""
    total = bound.json(authority/'counts.json')['candidate_training_histories']
    families = {source['ledger']: {} for source in sources}
    maximum = min(total, max(s['root_offset']+s['root_limit'] for s in sources))
    with (authority/'roots.jsonl').open() as stream:
        for index, line in enumerate(stream):
            if index >= maximum:
                break
            targets = [source for source in sources if source['root_offset'] <= index < source['root_offset']+source['root_limit']]
            if not targets:
                continue
            row = json.loads(line)
            history = replay(row['initial_fen'], row['moves'])
            if (row['split'] != 'train' or row['history'] != history or row['fen'] != history[-1] or
                    row['feature_key'] != history_key(history) or
                    row['id'] != 'recorded-search-'+row['feature_key']):
                raise ValueError('Selected original recorded root differs from its native history')
            for source in targets:
                families[source['ledger']][row['id']] = row
    if any(len(families[s['ledger']]) != s['original_roots'] for s in sources):
        raise ValueError('Native mining root-range coverage differs from its source manifest')
    return families


def _partitions(sources, work, chunk_rows):
    tasks = []
    for source in sources:
        start, offset, first, count, number, partial = 0, 0, 1, 0, 0, hashlib.sha256()
        with Path(source['ledger']).open('rb') as stream:
            for number, raw in enumerate(stream, 1):
                if not raw.endswith(b'\n') or not raw.strip():
                    raise ValueError('Oracle ledger must contain complete nonempty JSONL records')
                partial.update(raw); offset += len(raw); count += 1
                if count == chunk_rows:
                    tasks.append(dict(source, start=start, end=offset, first_line=first, records=count,
                        sha256=partial.hexdigest(), work=str(work/('part-%05d'%len(tasks)))))
                    start, first, count, partial = offset, number+1, 0, hashlib.sha256()
            if count:
                tasks.append(dict(source, start=start, end=offset, first_line=first, records=count,
                    sha256=partial.hexdigest(), work=str(work/('part-%05d'%len(tasks)))))
        if number != source['records']:
            raise ValueError('Native ledger coverage differs from its completed mining manifest')
    return tasks


class _Sink:
    def __init__(self, path, write):
        self.path, self.stream = path, path.open('xb') if write else None
        self.hash, self.bytes = hashlib.sha256(), 0

    def emit(self, value):
        raw = (_json(value)+'\n').encode()
        self.hash.update(raw); self.bytes += len(raw)
        if self.stream:
            self.stream.write(raw)

    def finish(self):
        if self.stream:
            self.stream.close()
        return {'sha256': self.hash.hexdigest(), 'bytes': self.bytes}


def _views(decision, numeric=True, move=True):
    key, query = decision['key'], decision['query']
    original = decision['record']
    base = dict(original, provenance='replayed_unrestricted_engine_query_ledger')
    common = dict(source_context_id=key, source_record_id=original['id'], source_query=query,
        full_teacher_explanation=False, history_before_initial_FEN_reconstructed=False,
        teacher=dict(provider='pinned_Pikafish', neural_prose_generated=False,
                     replayed_existing_engine_query=True, new_engine_query=False))
    rows = []
    if numeric:
        target = decision['target']
        rows.append(dict(base, id='oracle-numeric-'+key, stage=NUMERIC_STAGE,
            task_type='move_PV_candidates_evaluation_v1', question=NUMERIC_CHILD_QUESTION,
            answer=_json(target), future_moves=target['pv'], target_line_termination=decision['termination'],
            recorded_candidate_PV_positions_checked=decision['checked_positions'], **common))
    if move:
        row = move_label(dict(id='oracle-move-'+key, record=base, oracle={'best_move': decision['signature'][0]}))
        row.update(common)
        rows.append(row)
    return rows


def _merge(tasks, output, write, raw_counts, original_roots):
    first, signatures, conflicts, move_conflicts = {}, {}, set(), set()
    duplicates = 0
    for index, task in enumerate(tasks):
        with (Path(task['work'])/'decisions.jsonl').open() as stream:
            for number, line in enumerate(stream):
                row = json.loads(line); key, signature = row['key'], row['signature']
                if key in first:
                    duplicates += 1
                    if signatures[key] != signature:
                        conflicts.add(key)
                    if signatures[key][0] != signature[0]:
                        move_conflicts.add(key)
                else:
                    first[key], signatures[key] = (index, number), signature
    names = FILES[:6]
    sinks = {name: _Sink(output/name, write) for name in names}
    accepted, both, sides, positions, games = 0, 0, Counter(), set(), set()
    kinds, origins = Counter(), Counter()
    view_counts = Counter()
    excluded = {stage: Counter() for stage in [NUMERIC_STAGE, 'move_quality']}
    try:
        for index, task in enumerate(tasks):
            directory = Path(task['work'])
            with (directory/'excluded.jsonl').open() as stream:
                for line in stream:
                    sinks['rejected-queries.jsonl'].emit(json.loads(line))
            with (directory/'decisions.jsonl').open() as stream:
                for number, line in enumerate(stream):
                    row = json.loads(line); key = row['key']
                    if first[key] != (index, number):
                        sinks['duplicate-queries.jsonl'].emit(dict(context=key, query=row['query'],
                            signature=row['signature'], first_signature=signatures[key]))
                        continue
                    numeric_reason = ('repeated_context_has_conflicting_best_move_or_score' if key in conflicts
                                      else row['common_reason'] or row['numeric_reason'])
                    move_reason = ('repeated_context_has_conflicting_best_move' if key in move_conflicts
                                   else row['common_reason'])
                    for stage, reason in [(NUMERIC_STAGE, numeric_reason), ('move_quality', move_reason)]:
                        if reason is not None:
                            excluded[stage][reason] += 1
                            sinks['rejected-contexts.jsonl'].emit(dict(context=key, stage=stage,
                                query=row['query'], reason=reason))
                    if numeric_reason is not None and move_reason is not None:
                        continue
                    if numeric_reason is None and move_reason is not None:
                        raise ValueError('Numerical admission cannot bypass the common move contract')
                    for question in _views(row, numeric=numeric_reason is None, move=move_reason is None):
                        sinks['train.jsonl'].emit(question)
                        view_counts[question['stage']] += 1
                    accepted += 1; both += numeric_reason is None and move_reason is None
                    sides[row['record']['fen'].split()[1]] += 1
                    kinds[row['query']['source_root_kind']] += 1
                    origins['recorded_root' if row['query']['extra_plies_after_recorded_root'] == 0 else 'searched_descendant'] += 1
                    positions.add(position_key(row['record']['fen'])); games.add(row['record']['game_id'])
    finally:
        bindings = {str(output/name): sink.finish() for name, sink in sinks.items()}
    if not accepted:
        raise ValueError('No isolated numeric or move-answer contexts survive')
    counts = dict(sorted(raw_counts.items()))
    counts.update(original_search_roots_processed=original_roots,
        unique_potential_history_contexts=len(first), duplicate_query_records=duplicates,
        conflicting_best_move_contexts=len(move_conflicts), conflicting_best_move_or_score_contexts=len(conflicts),
        accepted_history_contexts=accepted, accepted_root_position_keys=len(positions),
        distinct_declared_source_game_ids=len(games), contexts_by_side=dict(sides),
        contexts_by_recorded_root_kind=dict(kinds), contexts_by_origin=dict(origins),
        rejected_history_contexts=len(first)-accepted,
        context_rejections_by_stage={stage: dict(values) for stage, values in excluded.items()},
        question_rows=sum(view_counts.values()), rows_by_stage={stage: view_counts[stage] for stage in excluded},
        shared_accepted_history_contexts=both,
        validation_rows=0, test_rows=0, new_color_derived_rows=0,
        full_teacher_explanations_created=0, new_engine_queries=0,
        student_training_executed=False, policy_or_coaching_gain_proven=False,
        root_and_all_recorded_candidate_PVs_and_color_mirrors_isolated=True,
        numeric_PV_six_plies_or_native_history_termination=True,
        question_views_counted_by_unique_history_union=True)
    if (raw_counts.get('native_checked_potential_records', 0) != len(first)+duplicates or
            any(view_counts[stage]+sum(excluded[stage].values()) != len(first) for stage in excluded)):
        raise ValueError('Query, context and derived-question coverage differs')
    return counts, bindings


def prepare(mining, authority, data, heldout_data, output, *, executable, weights,
            seed=20261013, minimum_nodes=100000, workers=8, chunk_rows=1000, resume=False):
    """Create both views, or fully replay a completed publication without rewriting it."""
    seed = checked_seed(seed)
    if (type(minimum_nodes) is not int or minimum_nodes < 100000 or
            type(workers) is not int or not 1 <= workers <= 32 or
            type(chunk_rows) is not int or chunk_rows < 1 or type(resume) is not bool or
            not isinstance(mining, (list, tuple)) or not mining):
        raise ValueError('Require native mining sources, >=100000 nodes and positive bounded CPU budgets')
    mining = [str(Path(path).resolve()) for path in mining]
    if len(set(mining)) != len(mining):
        raise ValueError('Mining sources must be distinct ordered directories')
    authority, data, heldout_data, output, executable, weights = map(lambda p: Path(p).resolve(),
        [authority, data, heldout_data, output, executable, weights])
    logical = dict(profile=PROFILE, mining=mining, authority=str(authority), data=str(data),
        heldout_data=str(heldout_data), output=str(output), executable=str(executable), weights=str(weights),
        seed=seed, minimum_nodes=minimum_nodes,
        questions_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in
                          [(NUMERIC_STAGE, NUMERIC_CHILD_QUESTION), ('move_quality', MOVE_QUESTION)]},
        first_query_precedence='first_natively_checked_unrestricted_query_in_source_order',
        query_origin='exact_recorded_root_range_or_legal_descendant_with_preserved_history',
        conflict_policies={NUMERIC_STAGE:'exclude_best_move_or_score_disagreement',
                           'move_quality':'exclude_best_move_disagreement'}, closure=_closure())
    saved, output_bound = None, BoundInputs()
    if output.exists():
        if not resume or not (output/'manifest.json').exists():
            raise FileExistsError('Preserve existing or partial output; use a fresh directory or completed --resume')
        saved = output_bound.json(output/'manifest.json')
        if saved.get('kind') != KIND or saved.get('status') != 'complete' or saved.get('config', {}).get('logical') != logical:
            raise ValueError('Completed policy-view source, schema or native contract changed')
        if {Path(p).resolve() for p in saved['outputs']} != {output/name for name in FILES}:
            raise ValueError('Policy publication output coverage changed')
        for name in FILES:
            output_bound.bind(output/name, _binding(saved['outputs'], output/name))
    elif resume:
        raise FileNotFoundError('Completed policy publication is required for --resume')
    bound = BoundInputs()
    reserved, common = shared_reservations(authority, data, heldout_data, seed, bound)
    sources = _sources(mining, authority, data, executable, weights, seed, bound)
    families = _root_families(sources, authority, bound)
    contract = dict(logical=logical, shared_isolation_authority=common, input_bindings=bound.artifacts)
    if saved is not None and (_read(output/'contract.json') != contract or saved['inputs'] != bound.artifacts):
        raise ValueError('Completed policy-view input identities changed')
    if saved is None:
        output.mkdir(parents=True)
        atomic_json(output/'contract.json', contract)
    work = Path(mkdtemp(prefix='oracle-policy-work-', dir=output.parent))
    try:
        tasks = _partitions(sources, work, chunk_rows)
        raw_counts = Counter()
        if workers == 1:
            for task in tasks:
                raw_counts.update(_worker(task, reserved, minimum_nodes, families))
        else:
            context = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=context,
                    initializer=_worker_init, initargs=(reserved, minimum_nodes, families)) as pool:
                pending = [pool.submit(_worker, task) for task in tasks]
                for future in as_completed(pending):
                    raw_counts.update(future.result())
        if raw_counts['raw_queries'] != sum(s['records'] for s in sources):
            raise ValueError('Native ledger row coverage differs')
        counts, bindings = _merge(tasks, output, saved is None, raw_counts,
                                   sum(s['original_roots'] for s in sources))
        bound.unchanged()
        if saved is not None:
            if counts != saved['verification'] or _read(output/'counts.json') != counts:
                raise ValueError('Replayed policy-view counts differ from their completed publication')
            for name, binding in bindings.items():
                if binding != _binding(saved['outputs'], name):
                    raise ValueError('Full native policy-view replay differs from published output bytes')
            output_bound.unchanged()
        else:
            atomic_json(output/'counts.json', counts)
            for name, expected in bindings.items():
                if {'sha256': digest(name), 'bytes': Path(name).stat().st_size} != expected:
                    raise ValueError('Published policy-view bytes changed after writing')
            result = manifest(KIND, dict(logical=logical, execution=dict(workers=workers,
                chunk_rows=chunk_rows, process_start_method='spawn' if workers > 1 else 'serial')),
                (), [output/name for name in FILES], counts)
            result['inputs'] = bound.artifacts
            bound.unchanged()
            atomic_json(output/'manifest.json', result)
    except BaseException:
        # Retain this attempt's own work for diagnosis; completed outputs are untouched.
        raise
    else:
        shutil.rmtree(work)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mining', nargs='+', required=True)
    for name in ['authority', 'data', 'heldout-data', 'output', 'executable', 'weights']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--minimum-nodes', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk-rows', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    with input_session_from_environment():
        print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
