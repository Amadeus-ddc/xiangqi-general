"""Bounded native mate/stalemate witnesses, with complete opponent replies.

This proves upper bounds after a fixed candidate, never shortest mate distances.
Full-history AXF outcomes stop search; rule-only wins are not mate witnesses.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

from . import rules
from .evidence import atomic_json, code_identity, digest, manifest

PROFILE = 'native_mate_stalemate_upper_bound_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _context(record):
    _require(isinstance(record, dict) and {'initial_fen', 'moves', 'fen'} <= record.keys(),
             'Require the original initial FEN, move list, and final FEN')
    initial, moves, fen = record['initial_fen'], record['moves'], record['fen']
    _require(isinstance(initial, str) and isinstance(fen, str) and
             isinstance(moves, list) and all(isinstance(m, str) for m in moves),
             'Require the original initial FEN, move list, and final FEN')
    _require(rules.replay(initial, moves)[-1] == fen, 'History does not end at the declared FEN')
    for i in range(len(moves)):
        _require(not rules.adjudicate(initial, moves[:i])['ended'],
                 'Original history continues beyond a terminal outcome')
    return {'initial_fen': initial, 'moves': list(moves), 'fen': fen}


def _budget(value):
    _require(type(value) is int and 0 <= value <= 8, 'Winner move bound must be an integer in [0, 8]')


class _Limit(Exception):
    pass


class _Search:
    def __init__(self, initial, winner, seconds, max_checks):
        self.initial, self.winner = initial, winner
        self.deadline, self.max_checks = time.monotonic() + seconds, max_checks
        self.checks = 0
        self.positions = lru_cache(maxsize=8192)(self._position)
        self.facts = lru_cache(maxsize=8192)(self._facts)

    def tick(self):
        self.checks += 1
        if self.checks > self.max_checks or time.monotonic() > self.deadline:
            raise _Limit

    def _position(self, history):
        self.tick()
        fen = rules.replay(self.initial, list(history))[-1]
        return fen, tuple(rules.legal_moves(fen))

    def _facts(self, history):
        fen, legal = self.positions(history)
        return fen, legal, rules.adjudicate(self.initial, list(history))

    def terminal(self, history):
        # A mate/stalemate witness requires zero legal replies. Defer the full
        # history ruling for other scan positions until solve expands them.
        fen, legal = self.positions(tuple(history))
        if legal:
            return None
        fen, legal, outcome = self.facts(tuple(history))
        if outcome['ended'] and outcome['winner'] == self.winner and not legal:
            return {'kind': 'terminal', 'upper': 0, 'fen': fen,
                    'terminal_kind': 'checkmate' if rules.in_check(fen) else 'stalemate_loss'}
        return None

    def solve(self, history, budget):
        self.tick()
        fen, legal, outcome = self.facts(tuple(history))
        if outcome['ended']:
            return self.terminal(history)
        if budget < 1:
            return None
        side = 'red' if fen.split()[1] == 'w' else 'black'
        if side == self.winner:
            for move in legal:  # Include quiet stalemate wins in the immediate scan.
                child = self.terminal(history + [move])
                if child is not None:
                    return {'kind': 'choose', 'move': move, 'child': child, 'upper': 1}
            if budget == 1:
                return None
            for move in sorted(legal, key=lambda m: not rules.gives_check(fen, m)):
                child = self.solve(history + [move], budget - 1)
                if child is not None:
                    return {'kind': 'choose', 'move': move, 'child': child, 'upper': child['upper'] + 1}
            return None
        children = []
        for move in legal:
            child = self.solve(history + [move], budget)
            if child is None:
                return None
            children.append({'reply': move, 'proof': child})
        if children:
            return {'kind': 'all', 'replies': list(legal), 'children': children,
                    'upper': max(c['proof']['upper'] for c in children)}
        return None


def prove_terminal_win(record, winner, *, max_winner_moves=3, max_state_checks=100000, seconds=45):
    """Return a positive certificate, bounded non-witness, or resource exhaustion.

    The budget counts the winner's moves starting at record['fen']. State checks
    count search/cache operations, not distinct chess positions. Inputs stay intact.
    """
    _budget(max_winner_moves)
    _require(winner in ('red', 'black'), 'Winner must be red or black')
    _require(type(max_state_checks) is int and max_state_checks > 0, 'State check limit must be positive')
    _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
             'Seconds must be finite and positive')
    context = _context(record)
    search = _Search(context['initial_fen'], winner, seconds, max_state_checks)
    proof = None
    started = time.monotonic()
    try:
        proof = search.solve(context['moves'], max_winner_moves)
        status = 'proved' if proof is not None else 'no_witness_within_bound'
    except _Limit:
        status = 'resource_exhausted'
    result = {'schema_version': 1, 'profile': PROFILE, 'status': status,
              'context_sha256': _hash(context), 'winner': winner,
              'max_winner_moves': max_winner_moves, 'max_state_checks': max_state_checks,
              'seconds_limit': seconds, 'state_checks': search.checks,
              'elapsed_seconds': time.monotonic() - started, 'proof': proof,
              'winner_moves_upper_bound': proof['upper'] if proof is not None else None,
              'shortest_mate_distance_proven': False}
    if proof is not None:
        verify_terminal_certificate(record, result)
    return result


def _verify(initial, history, winner, node, budget):
    _require(isinstance(node, dict), 'Malformed proof node')
    fen = rules.replay(initial, history)[-1]
    legal = list(rules.legal_moves(fen))
    outcome = rules.adjudicate(initial, history)
    if node.get('kind') == 'terminal':
        _require(not legal and outcome['ended'] and outcome['winner'] == winner,
                 'Proof leaf is not the required terminal win')
        _require(node.get('fen') == fen and type(node.get('upper')) is int and node['upper'] == 0,
                 'Changed terminal FEN or distance')
        kind = 'checkmate' if rules.in_check(fen) else 'stalemate_loss'
        _require(node.get('terminal_kind') == kind, 'Wrong terminal kind')
        return 0
    _require(not outcome['ended'] and budget >= 1, 'Proof continues beyond terminal or move bound')
    side = 'red' if fen.split()[1] == 'w' else 'black'
    if side == winner:
        _require(node.get('kind') == 'choose' and node.get('move') in legal, 'Illegal winner choice')
        upper = 1 + _verify(initial, history + [node['move']], winner, node.get('child'), budget - 1)
    else:
        children = node.get('children')
        _require(node.get('kind') == 'all' and node.get('replies') == legal and
                 isinstance(children, list) and len(children) == len(legal) and legal and
                 [c.get('reply') for c in children if isinstance(c, dict)] == legal,
                 'Proof must cover every native opponent reply exactly once')
        upper = max(_verify(initial, history + [c['reply']], winner, c.get('proof'), budget)
                    for c in children)
    _require(type(node.get('upper')) is int and node['upper'] == upper and upper <= budget,
             'Incorrect proved distance')
    return upper


def verify_terminal_certificate(record, certificate):
    """Read back a positive tree by re-enumerating all opponent replies."""
    context = _context(record)
    _require(isinstance(certificate, dict), 'Malformed certificate')
    _require(type(certificate.get('schema_version')) is int and certificate['schema_version'] == 1 and certificate.get('profile') == PROFILE and
             certificate.get('status') == 'proved', 'Require a supported positive certificate')
    _require(certificate.get('context_sha256') == _hash(context), 'Certificate history binding changed')
    winner, budget = certificate.get('winner'), certificate.get('max_winner_moves')
    _require(winner in ('red', 'black'), 'Invalid certificate winner')
    _budget(budget)
    upper = _verify(context['initial_fen'], context['moves'], winner, certificate.get('proof'), budget)
    _require(type(certificate.get('winner_moves_upper_bound')) is int and
             certificate['winner_moves_upper_bound'] == upper, 'Changed certificate distance')
    return upper


def _branch_context(record, branch):
    _require(isinstance(record, dict) and record.get('split') == 'train',
             'Candidate target audit requires an explicit train record')
    context = _context(record)
    _require(isinstance(branch, dict) and isinstance(branch.get('evaluation'), dict) and
             isinstance(branch.get('move'), str), 'Require a candidate move and evaluation')
    score, move = branch['evaluation'], branch['move']
    _require(score.get('type') == 'mate' and type(score.get('value')) is int and abs(score['value']) > 1
             and score.get('perspective') == 'side_to_move', 'Require a root-perspective branch mate estimate beyond one')
    _require(not rules.adjudicate(context['initial_fen'], context['moves'])['ended'] and
             move in rules.legal_moves(context['fen']), 'Fixed candidate must be legal at a nonterminal root')
    root_side = 'red' if context['fen'].split()[1] == 'w' else 'black'
    winner = root_side if score['value'] > 0 else ('black' if root_side == 'red' else 'red')
    after = dict(context, moves=context['moves'] + [move], fen=rules.play(context['fen'], move))
    return after, winner, int(winner == root_side)


def audit_mate_branch(record, branch, *, max_winner_moves=3, max_state_checks=100000, seconds=45):
    """Audit one branch estimate after its fixed move; root-global scores differ.

    A positive shorter upper bound contradicts that candidate's claimed distance.
    Neither a non-witness nor exhaustion certifies the original estimate as correct.
    """
    _budget(max_winner_moves)
    _require(max_winner_moves >= 1, 'Candidate audit needs a positive move bound')
    after, winner, offset = _branch_context(record, branch)
    bound = min(max_winner_moves, abs(branch['evaluation']['value']) - 1)
    certificate = prove_terminal_win(after, winner, max_winner_moves=bound - offset,
                                    max_state_checks=max_state_checks, seconds=seconds)
    positive = certificate['status'] == 'proved'
    return {'profile': PROFILE, 'status': 'positive_distance_contradiction' if positive else certificate['status'],
            'score_scope': 'fixed_candidate_root_perspective', 'branch_sha256': _hash(branch),
            'declared_evaluation': dict(branch['evaluation']), 'fixed': branch['move'],
            'fixed_winner_move_offset': offset, 'searched_winner_move_bound': bound,
            'proved_winner_moves_upper_bound': certificate['winner_moves_upper_bound'] + offset if positive else None,
            'positive_distance_contradiction': positive, 'original_score_certified_correct': False,
            'certificate': certificate}


def verify_mate_contradiction(record, branch, audit):
    after, winner, offset = _branch_context(record, branch)
    _require(isinstance(audit, dict), 'Malformed candidate audit')
    _require(audit.get('profile') == PROFILE and audit.get('status') == 'positive_distance_contradiction'
             and audit.get('positive_distance_contradiction') is True and
             audit.get('score_scope') == 'fixed_candidate_root_perspective' and
             audit.get('branch_sha256') == _hash(branch) and audit.get('fixed') == branch['move'] and
             audit.get('declared_evaluation') == branch['evaluation'] and
             type(audit.get('fixed_winner_move_offset')) is int and audit['fixed_winner_move_offset'] == offset,
             'Changed candidate audit binding')
    cert = audit.get('certificate')
    _require(isinstance(cert, dict) and cert.get('winner') == winner, 'Wrong candidate winner')
    upper = verify_terminal_certificate(after, cert) + offset
    _require(type(audit.get('proved_winner_moves_upper_bound')) is int and
             type(audit.get('searched_winner_move_bound')) is int and
             upper == audit['proved_winner_moves_upper_bound'] and
             cert['max_winner_moves'] + offset == audit.get('searched_winner_move_bound') and
             upper < abs(branch['evaluation']['value']), 'No verified shorter candidate distance')
    return upper


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSON object with record and branch')
    parser.add_argument('--output', type=Path, required=True, help='Fresh evidence output directory')
    parser.add_argument('--max-winner-moves', type=int, default=3)
    parser.add_argument('--max-state-checks', type=int, default=100000)
    parser.add_argument('--seconds', type=float, default=45)
    args = parser.parse_args(argv)
    _require(not args.output.exists(), 'Output already exists; preserve the previous attempt')
    before, code = digest(args.input), code_identity()
    value = json.loads(args.input.read_text())
    result = audit_mate_branch(value['record'], value['branch'], max_winner_moves=args.max_winner_moves,
                              max_state_checks=args.max_state_checks, seconds=args.seconds)
    if result['positive_distance_contradiction']:
        verify_mate_contradiction(value['record'], value['branch'], result)
    _require(digest(args.input) == before and code_identity() == code, 'Input or source changed during audit')
    args.output.mkdir(parents=True)
    output = args.output / 'audit.json'
    atomic_json(output, result)
    proof = manifest('native_terminal_branch_distance_audit',
        {'profile': PROFILE, 'max_winner_moves': args.max_winner_moves,
         'max_state_checks': args.max_state_checks, 'seconds': args.seconds},
        inputs=[args.input], outputs=[output], code=code,
        verification={'status': result['status'], 'positive_distance_contradiction': result['positive_distance_contradiction'],
                      'original_score_certified_correct': False, 'source_prose_or_targets_modified': False,
                      'model_or_engine_calls': 0, 'new_SGD_updates': 0, 'independent_rule_implementation': False})
    atomic_json(args.output / 'manifest.json', proof)
    print(json.dumps(proof['verification']))


if __name__ == '__main__':
    main()
