"""Mine improved model/child analyses; the oracle never supplies training prose."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from .calibration import candidate_probability, evaluation_probability
from .evidence import atomic_json, code_identity, digest, history_key, iter_jsonl, load_jsonl, manifest, position_key, write_jsonl
from .explanations import (EXPLANATION_QUESTION, continuation_positions, line_facts, move_facts,
                           parse_explanation, validate_explanation)
from .oracle import Pikafish
from .prose_moves import consolidation_messages
from .rules import adjudicate, legal_moves, piece_map, piece_name, play, replay, side
from .search_inputs import prepared_training_roots, training_roots
from .sampling import (SEED_PROFILE, checked_seed, generation_seed, generation_software,
                       search_generation_policy)
from .feature_store import feature_paths
from .search_critical import acceptance_evidence, checked_acceptance_policy, mine_paper


def first_divergence(fen, before, after):
    common = []
    for old, new in zip(before, after):
        if old != new:
            return fen, old, new, common
        fen = play(fen, old)
        common.append(old)
    return None


def descend(record, move):
    fen = play(record['fen'], move)
    history = record['history'] + [fen]
    return dict(record, id=record['id'] + '-' + move, fen=fen, history=history,
                moves=record['moves'] + [move], feature_key=history_key(history), future_moves=[],
                question=EXPLANATION_QUESTION)


def root_evaluation(child):
    score = child['evaluation']
    if score['perspective'] != 'side_to_move':
        raise ValueError('Child score has an incompatible perspective')
    value = -score['value']
    if score['type'] == 'mate' and value >= 0:
        value += 1
    return {'type': score['type'], 'value': value, 'perspective': 'side_to_move'}


def order_score(score):
    if score['type'] == 'mate':
        return (100000 - abs(score['value'])) * (1 if score['value'] > 0 else -1)
    return max(-99000, min(99000, score['value']))


def reserved_positions(data):
    positions = set()
    for split in ['validation', 'test']:
        for row in iter_jsonl(Path(data) / f'{split}.jsonl'):
            positions.add(position_key(row['fen']))
            for line in [row.get('future_moves', []), *row.get('future_branches', [])]:
                fen = row['fen']
                for move in line:
                    fen = play(fen, move); positions.add(position_key(fen))
            if row.get('stage') == 'explanation':
                positions.update(continuation_positions(row, parse_explanation(row['answer'])))
    return positions


def reachable_child_positions(record, value):
    """Check every reachable raw prefix, including unused lines that later fail."""
    if replay(record['initial_fen'], record['moves'])[-1] != record['fen']:
        raise ValueError('Child root differs from its complete history')
    positions = {position_key(record['fen'])}
    lines = [value.get('pv')]
    branches = value.get('branches')
    if isinstance(branches, list):
        lines.extend(b.get('pv') for b in branches if isinstance(b, dict))
    for line in lines:
        if not isinstance(line, list):
            continue
        fen, moves = record['fen'], list(record['moves'])
        for move in line:
            if not isinstance(move, str) or adjudicate(record['initial_fen'], moves)['ended']:
                break
            try:
                fen = play(fen, move)
            except (ValueError, TypeError):
                break
            moves.append(move)
            positions.add(position_key(fen))
    return positions


def checked_search_extensions(candidate_policy, reuse_child_analysis, max_root_retries):
    if candidate_policy not in ('strict', 'paper_fill') or type(reuse_child_analysis) is not bool:
        raise ValueError('Invalid candidate policy or child-analysis reuse setting')
    if (candidate_policy == 'paper_fill' or reuse_child_analysis) and max_root_retries is None:
        raise ValueError('Candidate filling and child reuse require an explicit numerical root retry policy')


def numerical_root_acceptance(row, generated):
    value = generated['analysis']
    try:
        if not isinstance(value, dict):
            raise ValueError('Root generation has no numerical evaluation')
        evaluation_probability(row['fen'], value.get('evaluation'))
        return {'valid': True, 'errors': []}
    except (AttributeError, TypeError, ValueError, OverflowError):
        return {'valid': False, 'errors': ['invalid_evaluation_contract']}


class SearchMiner:
    def __init__(self, predictor, oracle, reserved, nodes=100000, max_depth=5,
                 candidate_threshold=0.05, child_threshold=0.10, child_contract='full',
                 record_oracle_query=None, generation_policy=None, generation_seed=0,
                 max_root_retries=None, root_candidate_policy='strict', reuse_child_analysis=False,
                 acceptance_policy='strict', min_clean_ply=1):
        if child_contract not in {'full', 'move_eval'}:
            raise ValueError('Unknown child acceptance contract')
        self.predictor, self.oracle, self.reserved = predictor, oracle, reserved
        self.nodes, self.max_depth = nodes, max_depth
        self.candidate_threshold, self.child_threshold = candidate_threshold, child_threshold
        self.child_contract = child_contract
        self.engine_cache = {}
        self.record_oracle_query = record_oracle_query
        self.counts = Counter()
        self.generation_policy = search_generation_policy(generation_policy) if generation_policy is not None else None
        self.generation_seed = checked_seed(generation_seed) if self.generation_policy is not None else None
        self.generation_root, self.generation_index = None, 0
        if (max_root_retries is not None and (type(max_root_retries) is not int or max_root_retries < 0) or
                max_root_retries and self.generation_policy is None):
            raise ValueError('Root retries need a nonnegative integer budget and stochastic generation')
        self.max_root_retries = max_root_retries
        checked_search_extensions(root_candidate_policy, reuse_child_analysis, max_root_retries)
        self.root_candidate_policy, self.reuse_child_analysis = root_candidate_policy, reuse_child_analysis
        checked_acceptance_policy(acceptance_policy, root_candidate_policy, child_contract, max_root_retries, min_clean_ply)
        self.acceptance_policy, self.min_clean_ply = acceptance_policy, min_clean_ply

    def analyze(self, row, restricted=None):
        key = row['feature_key'], tuple(restricted or [])
        if key not in self.engine_cache:
            result = self.oracle.analyze(row['fen'], self.nodes, row['initial_fen'],
                                         row['moves'], searchmoves=restricted)
            if self.record_oracle_query:
                self.record_oracle_query({'record':{k:row[k] for k in
                    ['id', 'feature_key', 'game_id', 'split', 'initial_fen', 'moves', 'history', 'fen']},
                    'searchmoves':list(restricted or []), 'requested_nodes':self.nodes, 'response':result})
            self.engine_cache[key] = result
        return self.engine_cache[key]

    def move_probability(self, row, move, result):
        candidate = next((c for c in result['candidates'] if c['move'] == move), None)
        if candidate is None:
            result = self.analyze(row, [move])
            candidate = next(c for c in result['candidates'] if c['move'] == move)
        return candidate_probability(row['fen'], candidate)

    def model_analysis(self, row):
        generation = None
        if self.generation_policy is None:
            raw = self.predictor.generate(row, EXPLANATION_QUESTION, 768)
        else:
            source = self.generation_root if self.generation_root is not None else row
            seed = generation_seed(self.generation_seed, source, self.generation_index)
            generation = {'settings': self.generation_policy, 'seed_profile': SEED_PROFILE,
                          'seed': str(seed), 'call_index': self.generation_index}
            self.generation_index += 1
            self.counts['stochastic_model_generations'] += 1
            sampling = {k: v for k, v in self.generation_policy.items() if k != 'max_new_tokens'}
            raw = self.predictor.generate(row, EXPLANATION_QUESTION, self.generation_policy['max_new_tokens'],
                                          sampling=sampling, seed=seed)
        try:
            value = parse_explanation(raw)
            proof = validate_explanation(row['fen'], value)
            if proof['valid']:
                try:
                    continuation_positions(row, value)
                except ValueError:
                    proof['valid'] = False
                    proof['errors'].append('invalid_history_continuation')
        except (ValueError, TypeError, KeyError, IndexError):
            value, proof = None, {'valid': False, 'errors': ['invalid_json']}
        result = {'raw': raw, 'analysis': value, 'verification': proof}
        if generation is not None:
            result['generation'] = generation
        return result

    def root_analysis(self, row, cached_child=None):
        if cached_child is not None and not self.reuse_child_analysis:
            raise ValueError('Cached child reuse is not enabled for this search')
        if self.max_root_retries is None:
            return self.model_analysis(row), None
        attempts = []
        if cached_child is not None:
            fields = ['id', 'feature_key', 'initial_fen', 'moves', 'history', 'fen']
            if any(cached_child['record'][key] != row[key] for key in fields):
                raise ValueError('Cached child differs from the complete recursive root history')
            generated = copy.deepcopy({key: cached_child[key] for key in
                ['raw', 'analysis', 'verification', 'generation'] if key in cached_child})
            acceptance = numerical_root_acceptance(row, generated)
            generated['root_evaluation_acceptance'] = dict(acceptance, attempt=0)
            generated['reused_child_analysis'] = {'record_id': row['id'], 'feature_key': row['feature_key'],
                'source_child_acceptance': copy.deepcopy(cached_child['child_acceptance'])}
            attempts.append(generated)
            if acceptance['valid']:
                self.counts['reused_child_root_analyses'] += 1
                return generated, attempts
            self.counts['unusable_cached_child_evaluations'] += 1
        for index in range(self.max_root_retries + 1):
            generated = self.model_analysis(row)
            acceptance = numerical_root_acceptance(row, generated)
            generated['root_evaluation_acceptance'] = dict(acceptance, attempt=index + 1)
            attempts.append(generated)
            self.counts['root_evaluation_generations'] += 1
            if index:
                self.counts['root_evaluation_retries'] += 1
            if acceptance['valid']:
                break
        if not acceptance['valid']:
            self.counts['root_evaluation_retry_exhaustions'] += 1
        return generated, attempts

    def prepare_candidates(self, row, before, oracle):
        legal = legal_moves(row['fen'])
        required, candidates, sources = min(3, len(legal)), [], {}
        preferred = [before.get('move')]
        for field, values in [('preferred_move', preferred), ('candidates', before.get('candidates')),
                              ('original_pv', before.get('pv'))]:
            for move in values if isinstance(values, list) else []:
                if isinstance(move, str) and move in legal and move not in candidates and len(candidates) < required:
                    candidates.append(move); sources[move] = 'model_' + field
        model_candidates = list(candidates)
        filled = []
        for item in oracle['candidates']:
            move = item['move']
            if len(candidates) >= required:
                break
            if isinstance(move, str) and move in legal and move not in candidates:
                candidates.append(move); filled.append(move); sources[move] = 'engine_multipv_fill'
        if len(candidates) != required:
            raise ValueError('Engine supplied too few distinct legal root repair candidates')
        self.counts['engine_filled_root_candidates'] += len(filled)
        return candidates, {'policy': 'paper_fill', 'required': required,
            'legal_model_candidates': model_candidates, 'engine_filled_candidates': filled, 'sources': sources}

    def child_acceptance(self, row, generation):
        if self.child_contract == 'full':
            proof = generation['verification']
            return dict(proof, errors=list(proof['errors']), contract='full')
        value = generation['analysis']
        errors = []
        if not isinstance(value, dict):
            errors.append('invalid_json')
        else:
            move, candidates = value.get('move'), value.get('candidates')
            if not isinstance(move, str) or move not in legal_moves(row['fen']):
                errors.append('illegal_or_missing_move')
            elif not isinstance(candidates, list) or not candidates or candidates[0] != move:
                errors.append('inconsistent_preferred_move')
            try:
                evaluation_probability(row['fen'], value.get('evaluation'))
            except (AttributeError, TypeError, ValueError, OverflowError):
                errors.append('invalid_evaluation_contract')
        return {'valid': not errors, 'errors': errors, 'contract': 'move_eval'}

    def mine(self, original):
        if self.acceptance_policy == 'paper_recursive':
            return mine_paper(self, original)
        self.generation_root, self.generation_index = original, 0
        root = dict(original, question=EXPLANATION_QUESTION, future_moves=[])
        trace, cached_child = [], None
        for depth in range(self.max_depth + 1):
            self.counts['root_attempts'] += 1
            if position_key(root['fen']) in self.reserved:
                return None, trace, 'reserved_position'
            if adjudicate(root['initial_fen'], root['moves'])['ended']:
                return None, trace, 'terminal_root'
            generated, attempts = self.root_analysis(root, cached_child)
            cached_child = None
            step = {'record': root, 'depth': depth, 'root_generation': generated}
            trace.append(step)
            if attempts is not None:
                step['root_generation_attempts'] = attempts
                if not generated['root_evaluation_acceptance']['valid']:
                    return None, trace, 'invalid_root_evaluation'
            before = generated['analysis']
            candidates = before.get('candidates') if isinstance(before, dict) else None
            if self.root_candidate_policy == 'strict' and (not isinstance(candidates, list) or not 1 <= len(candidates) <= 3 or
                    any(not isinstance(m, str) or m not in legal_moves(root['fen']) for m in candidates) or
                    len(set(candidates)) != len(candidates)):
                return None, trace, 'invalid_root_candidates'
            oracle = self.analyze(root)
            if self.root_candidate_policy == 'paper_fill':
                candidates, preparation = self.prepare_candidates(root, before, oracle)
                step['candidate_preparation'] = preparation
            best = self.move_probability(root, oracle['best_move'], oracle)
            qualities = [(m, self.move_probability(root, m, oracle)) for m in candidates]
            injected = False
            if all(best - p >= self.candidate_threshold for _, p in qualities):
                worst = min(qualities, key=lambda x: x[1])[0]
                candidates = [m for m in candidates if m != worst] + [oracle['best_move']]
                injected = True; self.counts['oracle_candidate_injections'] += 1
                if self.root_candidate_policy == 'paper_fill':
                    preparation['sources'][oracle['best_move']] = 'engine_best_injection'
            if self.root_candidate_policy == 'paper_fill':
                candidates.sort(key=lambda move: self.move_probability(root, move, oracle), reverse=True)
                preparation['final_candidates'] = list(candidates)
            step.update(candidate_probabilities=qualities, injected_oracle_move=injected, oracle=oracle)
            children, offending, offending_generation = [], None, None
            for move in candidates:
                child = descend(root, move)
                if position_key(child['fen']) in self.reserved:
                    return None, trace, 'reserved_child'
                outcome = adjudicate(child['initial_fen'], child['moves'])
                if outcome['ended']:
                    root_color = 'red' if root['fen'].split()[1] == 'w' else 'black'
                    value = (1 if outcome['winner'] == root_color else -1) if outcome['winner'] else 0
                    evaluation = {'type': 'mate' if value else 'cp', 'value': value,
                                  'perspective': 'side_to_move'}
                    children.append({'move': move, 'record': child, 'outcome': outcome, 'analysis': None,
                                     'root_evaluation': evaluation, 'root_pv': [move]})
                    continue
                generation = self.model_analysis(child)
                acceptance = self.child_acceptance(child, generation)
                item = {'move': move, 'record': child, **generation, 'child_acceptance': acceptance}
                children.append(item)
                if not acceptance['valid']:
                    if offending is None:
                        offending, offending_generation = child, item
                    continue
                analysis = generation['analysis']
                positions = (continuation_positions(child, analysis) if self.child_contract == 'full'
                             else reachable_child_positions(child, analysis))
                if positions & self.reserved:
                    return None, trace, 'reserved_child_continuation'
                if not generation['verification']['valid']:
                    self.counts['full_child_contract_failures_bypassed_for_move_eval_check'] += 1
                engine = self.analyze(child)
                p_best = self.move_probability(child, engine['best_move'], engine)
                p_move = self.move_probability(child, analysis['move'], engine)
                p_model = evaluation_probability(child['fen'], analysis['evaluation'])
                pv = analysis.get('pv')
                inferred = ([move, *pv[:5]] if isinstance(pv, list) and pv and
                            pv[0] == analysis['move'] and all(isinstance(m, str) for m in pv[:5]) else None)
                item.update(oracle=engine, move_loss=max(0, p_best-p_move), evaluation_error=abs(p_model-p_best),
                            root_evaluation=root_evaluation(analysis), root_pv=inferred)
                if item['move_loss'] >= self.child_threshold or item['evaluation_error'] > self.child_threshold:
                    acceptance['valid'] = False
                    if item['move_loss'] >= self.child_threshold:
                        acceptance['errors'].append('recommended_move_mistake')
                    if item['evaluation_error'] > self.child_threshold:
                        acceptance['errors'].append('evaluation_error_exceeds_threshold')
                    if offending is None:
                        offending, offending_generation = child, item
            step['children'] = children
            if offending is not None:
                self.counts['recursive_descents'] += 1
                root = offending
                if self.reuse_child_analysis:
                    cached_child = offending_generation
                continue
            children.sort(key=lambda c: order_score(c['root_evaluation']), reverse=True)
            selected = children[0]
            selected_loss = max(0, best - self.move_probability(root, selected['move'], oracle))
            if selected_loss >= self.child_threshold:
                return None, trace, 'selected_root_move_still_mistake'
            if any(c['root_pv'] is None for c in children):
                return None, trace, 'invalid_inferred_continuation'
            inferred_lines = {'pv': selected['root_pv'],
                              'branches': [{'pv': c['root_pv']} for c in children]}
            try:
                inferred_positions = continuation_positions(root, inferred_lines)
            except (ValueError, TypeError):
                return None, trace, 'invalid_inferred_continuation'
            if inferred_positions & self.reserved:
                return None, trace, 'reserved_inferred_continuation'
            pv_before = before.get('pv')
            if not isinstance(pv_before, list) or not pv_before or any(not isinstance(m, str) for m in pv_before):
                return None, trace, 'invalid_original_pv'
            divergence = first_divergence(root['fen'], pv_before, selected['root_pv'])
            if divergence is None:
                return None, trace, 'no_measurable_pv_divergence'
            fen, old, new, prefix = divergence
            comparison = root
            for move in prefix:
                comparison = descend(comparison, move)
            if comparison['fen'] != fen or new not in legal_moves(fen):
                return None, trace, 'invalid_improved_pv'
            judge = self.analyze(comparison)
            p_new = self.move_probability(comparison, new, judge)
            p_old = self.move_probability(comparison, old, judge) if old in legal_moves(fen) else 0.0
            step['pv_divergence'] = {'before': old, 'after': new, 'before_probability': p_old,
                                     'after_probability': p_new, 'common_prefix': prefix}
            if p_new <= p_old:
                return None, trace, 'pv_does_not_strictly_improve'
            return self.consolidation_query(original, root, children, selected, oracle, selected_loss, depth), trace, 'accepted_for_consolidation'
        return None, trace, 'recursion_limit'


    def consolidation_query(self, original, root, children, selected, oracle, selected_loss, depth):
        best_candidate = next(c for c in oracle['candidates'] if c['move'] == oracle['best_move'])
        target = {'move': selected['move'], 'pv': selected['root_pv'],
                  'candidates': [c['move'] for c in children],
                  'branches': [{'move': c['move'], 'pv': c['root_pv'], 'evaluation': c['root_evaluation']}
                               for c in children],
                  'evaluation': {'type': best_candidate['score_type'], 'value': best_candidate['score'],
                                 'perspective': 'side_to_move'}, 'facts': move_facts(root['fen'], selected['move'])}
        content = {'root_side': side(root['fen']), 'root_fen': root['fen'],
                   'root_board': {square: piece_name(piece) for square, piece in piece_map(root['fen']).items()},
                   'target_fields': target,
                   'root_line_facts': [{'move': c['move'], 'facts': line_facts(root['fen'], c['root_pv'],
                                                                           include_positions=True)}
                                       for c in children],
                   'verified_child_analyses': [{'move': c['move'], 'analysis': c['analysis'],
                                                 'terminal_outcome': c.get('outcome')} for c in children]}
        if root.get('recorded_source_kind') in {'recorded_tactical_line', 'recorded_tactical_position'}:
            content['recorded_source_context'] = root['recorded_source_context']
        if self.child_contract == 'move_eval':
            del content['verified_child_analyses']
            content['child_analyses'] = [{'move': c['move'], 'terminal_outcome': c.get('outcome'),
                'recommended_move': c['analysis'].get('move') if c['analysis'] else None,
                'evaluation': c['analysis'].get('evaluation') if c['analysis'] else None,
                'used_pv': c['root_pv'][1:],
                'unverified_explanation': c['analysis'].get('explanation') if c['analysis'] else None,
                'full_output_verification': c.get('verification'),
                'move_evaluation_verification': c.get('child_acceptance')} for c in children]
        if self.acceptance_policy == 'paper_recursive':
            content['search_acceptance_policy'] = 'paper_recursive'
            for entry, child in zip(content['child_analyses'], children):
                sanitation = child['sanitation']
                entry['critical_sanitation'] = {key:sanitation.get(key) for key in
                    ['accepted_ply', 'accepted_uci', 'valid_checked_prefix', 'full_line_legal']}
        messages = consolidation_messages(content, root['fen'], target)
        self.counts['mined_improvements'] += 1
        return {'id': root['id'] + '-search', 'record': root, 'target_fields': target,
                'messages': messages, 'source_root_id': original['id'], 'depth': depth,
                'selected_root_move_loss': selected_loss,
                'child_contract': self.child_contract}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/research-balanced-v1')
    parser.add_argument('--features', default='data/research-balanced-v1/features-16.pt')
    pools = parser.add_mutually_exclusive_group()
    pools.add_argument('--prepared-inputs', help='Completed search_inputs directory bound to this data and seed')
    pools.add_argument('--recorded-inputs', help='Completed unused canonical-game roots and held-out positions')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=512)
    parser.add_argument('--nodes', type=int, default=100000)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--child-contract', choices=['full', 'move_eval'], default='full')
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--student-generation-config', help='Explicit sampling and output-token JSON; absent retains legacy greedy decoding')
    parser.add_argument('--max-root-retries', type=int,
                        help='Regenerate malformed numerical root evaluations; excludes the initial attempt')
    parser.add_argument('--root-candidate-policy', choices=['strict', 'paper_fill'],
                        help='Keep raw student output and optionally fill mining candidates from engine MultiPV')
    parser.add_argument('--reuse-child-analysis', action='store_true', default=None,
                        help='Reuse a numerically parsed child when that exact full-history position is recursed')
    parser.add_argument('--acceptance-policy', choices=['strict', 'paper_recursive'],
                        help='Explicit author-style clean-prefix, evaluation-improvement and recursive acceptance')
    parser.add_argument('--min-clean-ply', type=int, help='Minimum engine-checked child prefix; paper default is one')
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    args = parser.parse_args()
    if args.limit <= 0 or args.nodes <= 0 or args.max_depth < 0:
        raise ValueError('Search limits and node budgets must be positive; recursion depth cannot be negative')
    generation_policy = None
    if args.student_generation_config is not None:
        generation_policy = search_generation_policy(json.loads(Path(args.student_generation_config).read_text()))
        checked_seed(args.seed)
    if args.max_root_retries is not None and (args.max_root_retries < 0 or
            args.max_root_retries and generation_policy is None):
        raise ValueError('Root retries require a nonnegative budget and stochastic generation')
    checked_search_extensions(args.root_candidate_policy or 'strict', bool(args.reuse_child_analysis),
                              args.max_root_retries)
    checked_acceptance_policy(args.acceptance_policy or 'strict', args.root_candidate_policy or 'strict',
                              args.child_contract, args.max_root_retries, 1 if args.min_clean_ply is None else args.min_clean_ply)
    arguments = {k: v for k, v in vars(args).items()
                 if k not in {'student_generation_config', 'max_root_retries', 'root_candidate_policy',
                              'reuse_child_analysis', 'acceptance_policy', 'min_clean_ply'} or v is not None}
    root = Path(args.output)
    if (root / 'manifest.json').exists():
        raise FileExistsError('Completed search mining exists')
    from .inference import Predictor
    input_paths = [args.checkpoint, *feature_paths(args.features), args.executable, args.weights,
                   *[Path(args.data) / f'{s}.jsonl' for s in ['train', 'validation', 'test']]]
    if args.student_generation_config is not None:
        input_paths.append(Path(args.student_generation_config))
    if args.prepared_inputs:
        input_paths.extend(Path(args.prepared_inputs) / name for name in ['manifest.json', 'roots.jsonl', 'counts.json'])
    if args.recorded_inputs:
        from .recorded_search_inputs import ARTIFACTS, prepared_roots
        input_paths.extend(Path(args.recorded_inputs) / name for name in ARTIFACTS)
    contract = {'arguments': arguments, 'input_hashes': {str(p): digest(p) for p in input_paths},
                'code': code_identity()}
    if generation_policy is not None:
        contract['generation_software'] = generation_software()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'contract.json').exists():
        if json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Search continuation inputs, arguments or execution source changed')
    else:
        atomic_json(root / 'contract.json', contract)
    additional_reserved = set()
    if args.recorded_inputs:
        base_identities = {str(Path(args.data) / f'{s}.jsonl'): {
            'sha256': contract['input_hashes'][str(Path(args.data) / f'{s}.jsonl')],
            'bytes': (Path(args.data) / f'{s}.jsonl').stat().st_size}
            for s in ['train', 'validation', 'test']}
        records, additional_reserved = prepared_roots(args.recorded_inputs, args.data,
            args.limit, args.seed, base_identities)
    elif args.prepared_inputs:
        train = Path(args.data) / 'train.jsonl'
        records = prepared_training_roots(args.prepared_inputs, args.data, args.limit, args.seed,
            {'sha256': contract['input_hashes'][str(train)], 'bytes': train.stat().st_size})
    else:
        records = training_roots(args.data, args.limit, args.seed)
    partial = root / 'results.partial.jsonl'
    results = load_jsonl(partial) if partial.exists() else []
    oracle_log = root / 'oracle-queries.partial.jsonl'
    if results and not oracle_log.exists():
        raise ValueError('Original mining has no complete oracle ledger; resume its frozen source or use a fresh run')
    seen = {r['source_root_id'] for r in results}
    if len(seen) != len(results) or not seen <= {r['id'] for r in records}:
        raise ValueError('Search continuation IDs differ')
    predictor = Predictor(args.checkpoint, feature_cache=args.features)
    oracle = Pikafish(args.executable, args.weights, threads=2)
    try:
        with partial.open('a') as handle, oracle_log.open('a') as oracle_handle:
            def record_oracle_query(item):
                oracle_handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                oracle_handle.flush()

            miner = SearchMiner(predictor, oracle, reserved_positions(args.data) | additional_reserved,
                                args.nodes, args.max_depth,
                                child_contract=args.child_contract, record_oracle_query=record_oracle_query,
                                generation_policy=generation_policy, generation_seed=args.seed,
                                max_root_retries=args.max_root_retries,
                                root_candidate_policy=args.root_candidate_policy or 'strict',
                                reuse_child_analysis=bool(args.reuse_child_analysis),
                                acceptance_policy=args.acceptance_policy or 'strict',
                                min_clean_ply=1 if args.min_clean_ply is None else args.min_clean_ply)
            for row in records:
                if row['id'] in seen:
                    continue
                try:
                    query, trace, reason = miner.mine(row)
                except (RuntimeError, TimeoutError, ValueError) as error:
                    query, trace, reason = None, [{'error': str(error)}], 'mining_error'
                item = {'source_root_id': row['id'], 'reason': reason, 'query': query, 'trace': trace}
                handle.write(json.dumps(item, ensure_ascii=False) + '\n'); handle.flush()
                results.append(item)
                print(json.dumps({'processed': len(results), 'requested': len(records),
                                  'accepted': sum(r['query'] is not None for r in results)}), flush=True)
    finally:
        oracle.close()
    accepted = [r['query'] for r in results if r['query'] is not None]
    write_jsonl(root / 'queries.jsonl', accepted)
    proof = {'processed': len(results), 'accepted_for_consolidation': len(accepted),
             'reasons': dict(Counter(r['reason'] for r in results)), 'new_session_counters': dict(miner.counts),
             'training_roots_only': True, 'heldout_positions_excluded': True,
             'all_child_branch_positions_isolated': True, 'full_history_termination_checked': True,
             **acceptance_evidence(args.acceptance_policy or 'strict'),
             'probability_model': 'pinned_Pikafish_material_WDL',
             'teacher_consolidation_executed': False, 'student_generations_preserved': True,
             'child_contract': args.child_contract,
             'all_successful_oracle_queries_preserved': True,
             'restricted_move_queries_in_oracle_ledger': True,
             'oracle_query_records': sum(1 for _ in oracle_log.open()),
             'raw_child_reachable_prefixes_isolated': True,
             'all_inferred_target_lines_history_validated': True,
             'unused_child_fields_passed_to_consolidator': args.child_contract == 'full'}
    if args.recorded_inputs:
        proof.update(unused_recorded_training_inputs=True,
                     additional_reserved_canonical_and_explanation_positions=len(additional_reserved))
    if generation_policy is not None:
        proof.update(student_generation_policy=generation_policy, generation_seed_profile=SEED_PROFILE,
                     generation_software=contract['generation_software'],
                     stochastic_model_generations_this_session=miner.counts['stochastic_model_generations'])
    if args.max_root_retries is not None:
        proof.update(max_root_retries=args.max_root_retries,
                     root_retry_condition='malformed_numerical_evaluation',
                     root_evaluation_generations_this_session=miner.counts['root_evaluation_generations'],
                     root_evaluation_retries_this_session=miner.counts['root_evaluation_retries'],
                     root_evaluation_retry_exhaustions_this_session=miner.counts['root_evaluation_retry_exhaustions'])
    if args.root_candidate_policy is not None or args.reuse_child_analysis is not None:
        proof.update(root_candidate_policy=args.root_candidate_policy or 'strict',
                     reused_child_analysis_enabled=bool(args.reuse_child_analysis),
                     engine_filled_candidates_this_session=miner.counts['engine_filled_root_candidates'],
                     reused_child_root_analyses_this_session=miner.counts['reused_child_root_analyses'],
                     unusable_cached_child_evaluations_this_session=miner.counts['unusable_cached_child_evaluations'],
                     candidate_preparation_is_mining_only=True)
    if args.acceptance_policy is not None or args.min_clean_ply is not None:
        proof.update(acceptance_policy=args.acceptance_policy or 'strict',
                     min_clean_ply=1 if args.min_clean_ply is None else args.min_clean_ply,
                     raw_outputs_preserved=True, clean_prefixes_are_mining_only=True)
    atomic_json(root / 'manifest.json', manifest('search_distillation_mining', arguments, input_paths,
                [root / 'queries.jsonl', partial, oracle_log], proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
