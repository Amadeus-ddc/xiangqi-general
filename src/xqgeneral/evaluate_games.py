"""Full-history matches, including raw model forfeits and explicit censoring."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .evidence import atomic_json, code_identity, manifest, write_jsonl
from .explanations import EXPLANATION_QUESTION, continuation_positions, parse_explanation, validate_explanation
from .match_openings import checked_opening, load_openings
from .oracle import Pikafish
from .recorded_search_inputs import BoundInputs
from .rules import START_FEN, adjudicate, legal_moves, play, replay

OPENINGS = [('central_cannon', ['b2e2', 'h9g7', 'h0g2', 'b9c7']),
            ('two_horses', ['b0c2', 'b9c7', 'h0g2', 'h9g7'])]


def play_game(predictor, oracle, opening, model_color, nodes, max_plies, action_mode='explanation', move_beams=4):
    if (model_color not in {'red', 'black'} or action_mode not in {'explanation', 'legal_move'} or
            type(nodes) is not int or nodes <= 0 or type(max_plies) is not int or max_plies < 0 or
            type(move_beams) is not int or move_beams < 1):
        raise ValueError('Require a known model color/action mode and valid match budgets')
    if isinstance(opening, dict):
        checked_opening(opening)
        name, initial, book = opening['id'], opening['initial_fen'], opening['moves']
        provenance = {'opening_split': opening['split'], 'opening_game_id': opening['game_id'],
                      'opening_feature_key': opening['feature_key'],
                      'opening_source_declared_kind': opening.get('recorded_source_kind')}
    else:
        name, book = opening
        initial, provenance = START_FEN, {'opening_split': None}
    moves, history = list(book), replay(initial, book)
    context = {'opening': name, 'initial_fen': initial, 'opening_moves': list(book),
               'model_color': model_color, 'opponent_nodes': nodes, **provenance}
    turns = []
    for ply in range(max_plies + 1):
        outcome = adjudicate(initial, moves)
        if outcome['ended']:
            return {'status': 'completed', **outcome, 'moves': moves, 'turns': turns,
                    **context}
        if ply == max_plies:
            return {'status': 'censored', 'winner': None, 'reason': 'ply_limit', 'moves': moves,
                    'turns': turns, **context}
        fen = history[-1]
        mover = 'red' if fen.split()[1] == 'w' else 'black'
        if mover == model_color:
            record = {'initial_fen': initial, 'moves': list(moves), 'history': list(history), 'fen': fen,
                      'question': EXPLANATION_QUESTION}
            if action_mode == 'legal_move':
                raw = predictor.generate_moves([record], beams=move_beams)[0]
                move = raw
                proof = {'rule_legal_constraints': True, 'explanation_generated': False}
            elif action_mode == 'explanation':
                raw = predictor.generate(record, max_new_tokens=768)
                try:
                    analysis = parse_explanation(raw)
                    move = analysis.get('move')
                    proof = validate_explanation(fen, analysis, require_branches=True)
                    if proof['valid']:
                        try:
                            continuation_positions(record, analysis)
                        except ValueError:
                            proof['valid'] = False
                            proof['errors'].append('invalid_history_continuation')
                except (ValueError, TypeError, KeyError):
                    move, proof = None, {'valid': False, 'errors': ['invalid_json']}
            else:
                raise ValueError('Unknown match action mode')
            turns.append({'ply': len(moves), 'mover': mover, 'move': move, 'raw': raw, 'verification': proof,
                          'model_oracle_used': False, 'action_mode': action_mode})
            if not isinstance(move, str) or move not in legal_moves(fen):
                return {'status': 'completed', 'winner': 'black' if model_color == 'red' else 'red',
                        'reason': 'raw_model_invalid_move_forfeit', 'moves': moves, 'turns': turns,
                        **context}
        else:
            result = oracle.choose_move(fen, nodes, initial, moves)
            move = result['best_move']
            turns.append({'ply': len(moves), 'mover': mover, 'move': move, 'oracle': result})
        moves.append(move); history.append(play(fen, move))
    raise AssertionError('Match must terminate or be explicitly censored')


def match_summary(games):
    wins = sum(g['status'] == 'completed' and g['winner'] == g['model_color'] for g in games)
    losses = sum(g['status'] == 'completed' and g['winner'] not in {None, g['model_color']} for g in games)
    draws = sum(g['status'] == 'completed' and g['winner'] is None for g in games)
    censored = sum(g['status'] == 'censored' for g in games)
    completed = wins + losses + draws
    return {'games': len(games), 'wins': wins, 'losses': losses, 'draws': draws, 'censored': censored,
            'score_on_completed_games': (wins + .5*draws)/completed if completed else None,
            'termination_reasons': dict(Counter(g['reason'] for g in games)),
            'raw_model_forfeits': sum(g['reason'] == 'raw_model_invalid_move_forfeit' for g in games),
            'censored_games_counted_as_draws': False, 'elo_estimate': None, 'model_oracle_repairs': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--nodes', nargs='+', type=int, default=[100, 1000, 10000])
    parser.add_argument('--max-plies', type=int, default=256)
    parser.add_argument('--action-mode', choices=['explanation', 'legal_move'], default='explanation')
    parser.add_argument('--move-beams', type=int, default=4)
    parser.add_argument('--opening-suite', help='Completed heldout recorded opening suite directory')
    parser.add_argument('--expert-weights', default='models/px0-latest.pb.gz')
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh match evaluation output')
    if (args.max_plies <= 0 or args.move_beams < 1 or any(n <= 0 for n in args.nodes) or
            len(set(args.nodes)) != len(args.nodes)):
        raise ValueError('Match budgets must be positive and opponent node budgets distinct')
    openings, opening_proof = load_openings(args.opening_suite) if args.opening_suite else (OPENINGS, None)
    core = BoundInputs()
    for path in [args.checkpoint, args.executable, args.weights, args.expert_weights]:core.bind(path)
    if args.opening_suite:
        if core.json(Path(args.opening_suite) / 'manifest.json') != opening_proof:
            raise ValueError('Opening suite changed after admission')
        for group in ['inputs', 'outputs']:
            for path, item in opening_proof[group].items():core.bind(path, item)
    code = code_identity()
    from .inference import Predictor
    predictor = Predictor(args.checkpoint, args.expert_weights)
    oracle = Pikafish(args.executable, args.weights, threads=1, multipv=1)
    dest.mkdir(parents=True)
    started, games = time.monotonic(), []
    try:
        for nodes in args.nodes:
            for opening in openings:
                for color in ['red', 'black']:
                    game = play_game(predictor, oracle, opening, color, nodes, args.max_plies,
                                     args.action_mode, args.move_beams)
                    games.append(game)
                    atomic_json(dest/f'game-{len(games):03d}.json', game)
                    print(json.dumps(match_summary(games)), flush=True)
    finally:
        oracle.close()
    write_jsonl(dest/'games.jsonl', games)
    proof = {**match_summary(games), 'action_mode': args.action_mode,
             'openings': len(openings), 'opening_split': opening_proof['config']['split'] if opening_proof else None,
             'both_model_colors_at_each_opening_and_budget': True,
             'rule_legal_constraints': args.action_mode == 'legal_move',
             'explanation_quality_evaluated': False,
             'by_opponent_nodes': {str(n): match_summary([g for g in games if g['opponent_nodes'] == n])
                                                         for n in args.nodes},
             'seconds': time.monotonic()-started, 'rule_profile': 'pyffish-0.0.90-xiangqi-AXF'}
    atomic_json(dest/'metrics.json', proof)
    core.unchanged()
    if code_identity() != code:
        raise ValueError('Match execution source changed; preserve partial outputs')
    completed = manifest('raw_model_full_matches', vars(args), (),
                         [dest/'games.jsonl',dest/'metrics.json'], proof, code=code)
    completed['inputs'] = core.artifacts
    core.unchanged();atomic_json(dest/'manifest.json', completed)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
