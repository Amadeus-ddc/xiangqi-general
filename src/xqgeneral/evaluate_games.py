"""Full-history matches, including raw model forfeits and explicit censoring."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .evidence import atomic_json, digest, manifest, write_jsonl
from .explanations import EXPLANATION_QUESTION, parse_explanation, validate_explanation
from .oracle import Pikafish
from .rules import START_FEN, adjudicate, legal_moves, play, replay

OPENINGS = [('central_cannon', ['b2e2', 'h9g7', 'h0g2', 'b9c7']),
            ('two_horses', ['b0c2', 'b9c7', 'h0g2', 'h9g7'])]


def play_game(predictor, oracle, opening, model_color, nodes, max_plies, action_mode='explanation', move_beams=4):
    name, book = opening
    moves, history = list(book), replay(START_FEN, book)
    turns = []
    for ply in range(max_plies + 1):
        outcome = adjudicate(START_FEN, moves)
        if outcome['ended']:
            return {'status': 'completed', **outcome, 'moves': moves, 'turns': turns,
                    'opening': name, 'model_color': model_color, 'opponent_nodes': nodes}
        if ply == max_plies:
            return {'status': 'censored', 'winner': None, 'reason': 'ply_limit', 'moves': moves,
                    'turns': turns, 'opening': name, 'model_color': model_color, 'opponent_nodes': nodes}
        fen = history[-1]
        mover = 'red' if fen.split()[1] == 'w' else 'black'
        if mover == model_color:
            record = {'initial_fen': START_FEN, 'moves': list(moves), 'history': list(history), 'fen': fen,
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
                except (ValueError, TypeError, KeyError):
                    move, proof = None, {'valid': False, 'errors': ['invalid_json']}
            else:
                raise ValueError('Unknown match action mode')
            turns.append({'ply': len(moves), 'mover': mover, 'move': move, 'raw': raw, 'verification': proof,
                          'model_oracle_used': False, 'action_mode': action_mode})
            if not isinstance(move, str) or move not in legal_moves(fen):
                return {'status': 'completed', 'winner': 'black' if model_color == 'red' else 'red',
                        'reason': 'raw_model_invalid_move_forfeit', 'moves': moves, 'turns': turns,
                        'opening': name, 'model_color': model_color, 'opponent_nodes': nodes}
        else:
            result = oracle.choose_move(fen, nodes, START_FEN, moves)
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
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh match evaluation output')
    if args.max_plies <= 0 or args.move_beams < 1 or any(n <= 0 for n in args.nodes):
        raise ValueError('Match budgets must be positive')
    from .inference import Predictor
    predictor = Predictor(args.checkpoint)
    oracle = Pikafish(args.executable, args.weights, threads=1, multipv=1)
    dest.mkdir(parents=True)
    started, games = time.monotonic(), []
    try:
        for nodes in args.nodes:
            for opening in OPENINGS:
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
             'rule_legal_constraints': args.action_mode == 'legal_move',
             'explanation_quality_evaluated': False,
             'by_opponent_nodes': {str(n): match_summary([g for g in games if g['opponent_nodes'] == n])
                                                         for n in args.nodes},
             'seconds': time.monotonic()-started, 'rule_profile': 'pyffish-0.0.90-xiangqi-AXF'}
    atomic_json(dest/'metrics.json', proof)
    atomic_json(dest/'manifest.json', manifest('raw_model_full_matches', vars(args),
                [args.checkpoint,args.executable,args.weights], [dest/'games.jsonl',dest/'metrics.json'], proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
