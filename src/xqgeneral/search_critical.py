"""Full-history critical-line auditing and the paper's recursive decisions."""
from .rules import adjudicate, legal_moves, replay


def checked_acceptance_policy(policy, candidate_policy, child_contract, root_retries, min_clean_ply):
    if policy not in {'strict', 'paper_recursive'} or type(min_clean_ply) is not int or min_clean_ply < 1:
        raise ValueError('Invalid search acceptance policy or clean-prefix length')
    if policy == 'paper_recursive' and (candidate_policy != 'paper_fill' or
            child_contract != 'move_eval' or root_retries is None):
        raise ValueError('Paper recursion requires candidate filling, numerical roots and move_eval children')
    if policy == 'strict' and min_clean_ply != 1:
        raise ValueError('A clean-prefix length requires the paper recursive policy')


def parse_critical(record, moves):
    from .search_distillation import descend
    if replay(record['initial_fen'], record['moves']) != record['history'] or record['history'][-1] != record['fen']:
        raise ValueError('Critical-line root differs from its complete history')
    emitted = moves if isinstance(moves, list) else []
    current, steps, illegal = record, [], None
    terminal = adjudicate(current['initial_fen'], current['moves'])['ended']
    for index, move in enumerate(emitted, 1):
        if terminal:
            break
        if not isinstance(move, str) or move not in legal_moves(current['fen']):
            illegal = {'ply': index, 'reason': 'illegal', 'token_text': move}
            break
        steps.append({'ply': index, 'fen': current['fen'], 'uci': move, 'record': current})
        current = descend(current, move)
        terminal = adjudicate(current['initial_fen'], current['moves'])['ended']
    if not emitted and not terminal:
        illegal = {'ply': 1, 'reason': 'unparseable_or_empty'}
    return {'parsed_move_count': len(emitted), 'legal_steps': steps, 'illegal_at': illegal, 'terminal': terminal}


def finish_critical(parsed, audits, mistake_drop=0.10, min_clean_ply=1):
    plies = [dict(step, **audits[(step['record']['feature_key'], step['uci'])])
             for step in parsed['legal_steps']]
    mistake = next((dict(p, reason='mistake') for p in plies if p['winrate_drop'] >= mistake_drop), None)
    cut = min([len(plies)] + [p['ply'] - 1 for p in [parsed['illegal_at'], mistake] if p is not None])
    emitted = parsed['parsed_move_count']
    return {'parsed_move_count': emitted, 'accepted_ply': cut,
        'accepted_uci': [p['uci'] for p in plies[:cut]], 'ply_audit': plies,
        'illegal_at': parsed['illegal_at'], 'first_mistake': mistake,
        'full_line_legal': parsed['illegal_at'] is None and (emitted > 0 or parsed['terminal']),
        'valid_checked_prefix': (emitted >= min_clean_ply and cut >= min_clean_ply) or
                               (parsed['terminal'] and cut == len(plies))}


def parsed_sequence(parsed):
    result = [{'uci': step['uci'], 'legal': True} for step in parsed['legal_steps']]
    if parsed['illegal_at'] is not None:
        result.append({'uci': None, 'legal': False, 'token_text': parsed['illegal_at'].get('token_text')})
    return result


def comparison_boundary(sanitation):
    removed = [p for p in [sanitation['illegal_at'], sanitation['first_mistake']] if p is not None]
    if not removed:
        return None
    first = min(removed, key=lambda p: p['ply'])
    if first['reason'] == 'mistake':
        return {'uci': first['uci'], 'legal': True, 'rejected': 'mistake', 'winrate_drop': first['winrate_drop']}
    return {'uci': None, 'legal': False, 'rejected': 'illegal', 'token_text': first.get('token_text')}


def first_critical_divergence(record, old, new):
    from .search_distillation import descend
    current = record
    for index, (before, after) in enumerate(zip(old, new), 1):
        if (before.get('uci'), before['legal']) != (after.get('uci'), after['legal']):
            return {'kind': 'divergence', 'ply': index, 'fen': current['fen'],
                    'record': current, 'old': before, 'new': after}
        if not before['legal']:
            return {'kind': 'identical_illegal', 'ply': index, 'old': before, 'new': after}
        current = descend(current, before['uci'])
    return {'kind': 'no_divergence', 'shared_ply': min(len(old), len(new)),
            'old_length': len(old), 'new_length': len(new)}


def child_order(evaluation):
    value = evaluation['value']
    if evaluation['type'] == 'mate':
        return (-1, value) if value > 0 else (1, value)
    return (0, -value)


def decide(children, original_probability, truth, divergence, inject_drop=0.05, eval_truth_limit=0.10):
    """Children arrive in oracle-quality order; numerical selection preserves ties."""
    def recurse(child, reason, **details):
        return dict(details, action='recurse', reason=reason,
                    recurse_record_id=child['record']['id'], recurse_feature_key=child['record']['feature_key'])
    malformed = next((c for c in children if 'predicted_root_probability' not in c), None)
    if malformed is not None:
        return recurse(malformed, 'child evaluation remained malformed')
    selected = max(children, key=lambda c: c['predicted_root_order'])
    best = children[0]
    inferred = selected['predicted_root_probability']
    details = {'selected_move': selected['move'], 'e_prime': inferred,
               'new_eval_error': abs(inferred - truth), 'old_eval_error': abs(original_probability - truth)}
    if selected is not best and best['true_root_probability'] - selected['true_root_probability'] >= inject_drop:
        offending = max(children, key=lambda c: abs(c['predicted_root_probability'] - c['true_root_probability']))
        return recurse(offending, 'incorrect inaccurate child ordering', **details)
    if details['new_eval_error'] >= eval_truth_limit:
        return recurse(best, 'inferred evaluation outside truth limit', **details)
    unusable = next((c for c in children if not c['sanitation']['valid_checked_prefix']), None)
    if unusable is not None:
        return recurse(unusable, 'child exhausted clean-prefix attempts', **details)
    if divergence['accept']:
        return dict(details, action='accept', reason=divergence['reason'])
    return recurse(selected, divergence['reason'], **details)


def compare_lines(miner, root, before, selected, truth):
    old = parsed_sequence(parse_critical(root, before.get('pv')))
    new = [{'uci': m, 'legal': True} for m in selected['root_pv']]
    divergence = first_critical_divergence(root, old, new)
    boundary = comparison_boundary(selected['sanitation'])
    if divergence['kind'] == 'no_divergence' and len(new) < len(old) and boundary is not None:
        divergence = first_critical_divergence(root, old, [*new, boundary])
        divergence['used_truncation_boundary'] = True
    kind = divergence['kind']
    if kind == 'divergence':
        previous, proposed = divergence['old'], divergence['new']
        if previous['legal'] != proposed['legal']:
            accepted = proposed['legal']
            reason = 'new line repairs illegal first divergence' if accepted else 'new line becomes illegal at first divergence'
        elif not previous['legal']:
            accepted, reason = False, 'both lines are illegal at first divergence'
        else:
            record = divergence['record']
            # Both moves share one root, node budget, complete history and restricted query.
            engine = miner.analyze(record, [previous['uci'], proposed['uci']])
            scores = {c['move']: c for c in engine['candidates']}
            from .search_distillation import order_score
            old_score = order_score({'type':scores[previous['uci']]['score_type'], 'value':scores[previous['uci']]['score']})
            new_score = order_score({'type':scores[proposed['uci']]['score_type'], 'value':scores[proposed['uci']]['score']})
            divergence.update(old_score=old_score, new_score=new_score)
            accepted = new_score > old_score
            reason = 'new move is better at first divergence' if accepted else 'new move is not better at first divergence'
    elif kind == 'identical_illegal':
        accepted, reason = False, 'same illegal line'
    else:
        from .calibration import evaluation_probability
        old_error = abs(evaluation_probability(root['fen'], before['evaluation']) - truth)
        new_error = abs(selected['predicted_root_probability'] - truth)
        divergence.update(old_eval_error=old_error, new_eval_error=new_error)
        accepted = new_error < old_error
        reason = 'prefix-equivalent line has a closer evaluation' if accepted else 'prefix-equivalent line does not improve evaluation'
    divergence.update(accept=accepted, reason=reason)
    return divergence


def mine_paper(miner, original):
    from .calibration import evaluation_probability
    from .explanations import EXPLANATION_QUESTION
    from .evidence import position_key
    from .search_distillation import descend, reachable_child_positions, root_evaluation
    miner.generation_root, miner.generation_index = original, 0
    root, cached, trace = dict(original, question=EXPLANATION_QUESTION, future_moves=[]), None, []
    for depth in range(miner.max_depth + 1):
        miner.counts['root_attempts'] += 1
        if position_key(root['fen']) in miner.reserved:
            return None, trace, 'reserved_position'
        if adjudicate(root['initial_fen'], root['moves'])['ended']:
            return None, trace, 'terminal_root'
        generated, attempts = miner.root_analysis(root, cached)
        step = {'record':root, 'depth':depth, 'root_generation':generated, 'root_generation_attempts':attempts}
        trace.append(step)
        if not generated['root_evaluation_acceptance']['valid']:
            return None, trace, 'invalid_root_evaluation'
        before = generated['analysis']
        if reachable_child_positions(root, before) & miner.reserved:
            return None, trace, 'reserved_original_continuation'
        oracle = miner.analyze(root)
        candidates, preparation = miner.prepare_candidates(root, before, oracle)
        truth = miner.move_probability(root, oracle['best_move'], oracle)
        qualities = {m:miner.move_probability(root,m,oracle) for m in candidates}
        injected = oracle['best_move'] not in candidates and all(truth-p >= miner.candidate_threshold for p in qualities.values())
        if injected:
            worst=min(candidates,key=qualities.get)
            candidates=[m for m in candidates if m!=worst]+[oracle['best_move']]
            qualities[oracle['best_move']]=truth
            preparation['sources'][oracle['best_move']]='engine_best_injection'
            miner.counts['oracle_candidate_injections']+=1
        candidates.sort(key=qualities.get,reverse=True)
        preparation['final_candidates']=list(candidates)
        step.update(oracle=oracle,candidate_preparation=preparation,
                    candidate_probabilities=list(qualities.items()),injected_oracle_move=injected)
        children=[]
        step['children']=children  # Retain completed raw children even on an early reservation rejection.
        for move in candidates:
            child=descend(root,move)
            if position_key(child['fen']) in miner.reserved:
                return None,trace,'reserved_child'
            outcome=adjudicate(child['initial_fen'],child['moves'])
            item={'move':move,'record':child,'true_root_probability':qualities[move]}
            children.append(item)
            if outcome['ended']:
                winner=outcome['winner'];root_color='red' if root['fen'].split()[1]=='w' else 'black'
                value=(1 if winner==root_color else -1) if winner else 0
                item.update(outcome=outcome,analysis=None,root_evaluation={'type':'mate' if value else 'cp',
                    'value':value,'perspective':'side_to_move'},root_pv=[move],
                    predicted_root_probability=(value+1)/2,predicted_root_order=(value,0),
                    sanitation={'valid_checked_prefix':True,'accepted_uci':[],'accepted_ply':0,
                                'illegal_at':None,'first_mistake':None,'terminal_outcome':outcome})
                continue
            child_generation=miner.model_analysis(child)
            item.update(child_generation,child_acceptance=miner.child_acceptance(child,child_generation))
            analysis=child_generation['analysis']
            if isinstance(analysis,dict) and reachable_child_positions(child,analysis) & miner.reserved:
                return None,trace,'reserved_child_continuation'
            parsed=parse_critical(child,analysis.get('pv') if isinstance(analysis,dict) else None)
            audits={}
            for ply in parsed['legal_steps']:
                current=ply['record'];engine=miner.analyze(current)
                p_best=miner.move_probability(current,engine['best_move'],engine)
                p_move=miner.move_probability(current,ply['uci'],engine)
                audits[(current['feature_key'],ply['uci'])]={'winrate_drop':max(0,p_best-p_move),
                    'best_probability':p_best,'move_probability':p_move}
            sanitation=finish_critical(parsed,audits,miner.child_threshold,miner.min_clean_ply)
            item.update(sanitation=sanitation,root_pv=[move,*sanitation['accepted_uci']])
            from .search_distillation import numerical_root_acceptance
            numerical=numerical_root_acceptance(child,child_generation)
            item['numerical_acceptance']=numerical
            if numerical['valid']:
                item.update(predicted_root_probability=1-evaluation_probability(child['fen'],analysis['evaluation']),
                    predicted_root_order=child_order(analysis['evaluation']),root_evaluation=root_evaluation(analysis))
        step['children']=children
        usable=all('predicted_root_probability' in c and c['sanitation']['valid_checked_prefix'] for c in children)
        selected=max(children,key=lambda c:c['predicted_root_order']) if all('predicted_root_order' in c for c in children) else None
        divergence=compare_lines(miner,root,before,selected,truth) if usable else None
        step['paper_divergence']=divergence
        decision=decide(children,evaluation_probability(root['fen'],before['evaluation']),truth,divergence,
                        miner.candidate_threshold,miner.child_threshold)
        step['paper_decision']=decision
        if decision['action']=='recurse':
            offending=next(c for c in children if c['record']['feature_key']==decision['recurse_feature_key'])
            miner.counts['recursive_descents']+=1
            root=offending['record']
            cached=offending if miner.reuse_child_analysis and 'raw' in offending else None
            continue
        if any(len(c['root_pv'])>32 for c in children):
            return None,trace,'target_pv_exceeds_structured_contract'
        selected_loss=max(0,truth-selected['true_root_probability'])
        rendered=sorted(children,key=lambda c:c['predicted_root_order'],reverse=True)
        query=miner.consolidation_query(original,root,rendered,selected,oracle,selected_loss,depth)
        query['acceptance_policy']='paper_recursive'
        return query,trace,'accepted_for_consolidation'
    return None,trace,'recursion_limit'
