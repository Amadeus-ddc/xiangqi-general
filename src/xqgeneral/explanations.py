"""Machine-checkable explanations and root/child perspective conversion."""
import json
import math
import re
from .rules import gives_check, legal_moves, play, piece_map, piece_name

EXPLANATION_QUESTION = ('请分析当前局面，以JSON回答，包含move（建议走法）、pv（主要变化列表）、'
                        'candidates（三个候选走法列表）、branches（各候选的move、pv和evaluation）、'
                        'evaluation（type为cp或mate、value为整数、'
                        'perspective为side_to_move）、facts（建议走法的piece/from/to/captured/check事实）、'
                        'explanation（解释选招理由、备选差异与风险的中文讲解）。')


def move_facts(fen, move):
    if move not in legal_moves(fen):
        raise ValueError(f'Illegal factual move {move}')
    board = piece_map(fen)
    return {'piece': piece_name(board[move[:2]]), 'from': move[:2], 'to': move[2:],
            'captured': piece_name(board.get(move[2:])), 'check': gives_check(fen, move)}


def line_facts(fen, moves):
    result = []
    for move in moves:
        result.append({'move': move, **move_facts(fen, move)})
        fen = play(fen, move)
    return result


def parse_explanation(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Explanation must be a JSON object')
    return value


def validate_explanation(fen, value, require_facts=True):
    if not isinstance(value, dict):
        return {'valid': False, 'errors': ['not_an_object'],
                'strategic_prose_verdict': 'requires_separate_human_review'}
    errors = []
    move = value.get('move')
    legal = legal_moves(fen)
    if move not in legal:
        errors.append('illegal_or_missing_move')
    pv = value.get('pv')
    if not isinstance(pv, list) or not pv or pv[0] != move:
        errors.append('pv_must_begin_with_move')
    elif len(pv) > 32:
        errors.append('pv_too_long')
    else:
        position = fen
        for m in pv:
            try:
                position = play(position, m)
            except (ValueError, TypeError):
                errors.append('illegal_pv')
                break
    candidates = value.get('candidates')
    if (not isinstance(candidates, list) or not candidates or len(candidates) > 4 or
            any(not isinstance(m, str) or m not in legal for m in candidates) or
            len(set(candidates)) != len(candidates) or candidates[0] != move):
        errors.append('invalid_candidates')
    def valid_score(score):
        return (isinstance(score, dict) and score.get('type') in {'cp', 'mate'} and
                type(score.get('value')) is int and score.get('perspective') == 'side_to_move')
    if not valid_score(value.get('evaluation')):
        errors.append('invalid_evaluation_contract')
    branch_moves = []
    branches = value.get('branches', [])
    if not isinstance(branches, list) or len(branches) > 4:
        errors.append('invalid_branches')
        branches = []
    for branch in branches:
        if (not isinstance(branch, dict) or not isinstance(branch.get('pv'), list) or
                not branch['pv'] or len(branch['pv']) > 32 or branch['pv'][0] != branch.get('move') or
                not valid_score(branch.get('evaluation'))):
            errors.append('invalid_branch_contract')
            continue
        branch_moves.append(branch['move'])
        position = fen
        for m in branch['pv']:
            try:
                position = play(position, m)
            except (ValueError, TypeError):
                errors.append('illegal_branch_pv')
                break
    if branches and (branch_moves != candidates or len(set(branch_moves)) != len(branch_moves)):
        errors.append('branch_candidates_mismatch')
    if require_facts and move in legal and value.get('facts') != move_facts(fen, move):
        errors.append('incorrect_move_facts')
    prose = value.get('explanation')
    if not isinstance(prose, str) or not re.search(r'[\u4e00-\u9fff]', prose):
        errors.append('missing_chinese_explanation')
    elif isinstance(pv, list) and isinstance(candidates, list):
        allowed = {m for m in pv + candidates if isinstance(m, str)}
        allowed.update(m for b in branches if isinstance(b, dict) and isinstance(b.get('pv'), list)
                       for m in b['pv'] if isinstance(m, str))
        if any(m not in allowed for m in re.findall(r'[a-i][0-9][a-i][0-9]', prose)):
            errors.append('unsupported_prose_move')
    return {'valid': not errors, 'errors': errors, 'strategic_prose_verdict': 'requires_separate_human_review'}


def expected_score(candidate):
    if candidate.get('wdl'):
        win, draw, loss = candidate['wdl']
        total = win + draw + loss
        if total <= 0:
            raise ValueError('Invalid engine WDL')
        return (win + draw * 0.5) / total
    if candidate['score_type'] == 'mate':
        return 1.0 if candidate['score'] > 0 else 0.0
    # Only a monotone diagnostic when the engine does not emit calibrated WDL.
    return 1 / (1 + math.exp(-max(-10000, min(10000, candidate['score'])) / 300))


def root_score(child):
    evaluation = child['evaluation']
    if evaluation['perspective'] != 'side_to_move':
        raise ValueError('Child evaluation must be from the child mover perspective')
    value = evaluation['value']
    if evaluation['type'] == 'mate':
        return (-1 if value > 0 else 1) * (100000 - abs(value))
    if evaluation['type'] == 'cp':
        return -value
    raise ValueError('Unknown evaluation unit')


def choose_child(children):
    if not children:
        raise ValueError('Cannot consolidate an empty branch set')
    return max(children, key=lambda c: root_score(c['analysis']))


def oracle_explanation(fen, result, prose):
    best = result['best_move']
    selected = next(c for c in result['candidates'] if c['move'] == best)
    moves = [best] + [c['move'] for c in result['candidates'] if c['move'] != best]
    value = {'move': best, 'pv': selected['pv'][:6], 'candidates': moves[:3],
             'evaluation': {'type': selected['score_type'], 'value': selected['score'], 'perspective': 'side_to_move'},
             'facts': move_facts(fen, best), 'explanation': prose}
    by_move = {c['move']: c for c in result['candidates']}
    value['branches'] = [{'move': m, 'pv': by_move[m]['pv'][:6],
                          'evaluation': {'type': by_move[m]['score_type'], 'value': by_move[m]['score'],
                                         'perspective': 'side_to_move'}} for m in value['candidates']]
    proof = validate_explanation(fen, value)
    if not proof['valid']:
        raise ValueError(f'Invalid labeled explanation: {proof["errors"]}')
    return value
