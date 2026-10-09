"""Full-history matches, including raw model forfeits and explicit censoring."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
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


def _context(opening, model_color, nodes):
    if isinstance(opening, dict):
        checked_opening(opening)
        name, initial, book = opening['id'], opening['initial_fen'], opening['moves']
        provenance = {'opening_split': opening['split'], 'opening_game_id': opening['game_id'],
                      'opening_feature_key': opening['feature_key'],
                      'opening_source_declared_kind': opening.get('recorded_source_kind')}
    else:
        name, book = opening
        initial, provenance = START_FEN, {'opening_split': None}
    return {'opening': name, 'initial_fen': initial, 'opening_moves': list(book),
            'model_color': model_color, 'opponent_nodes': nodes, **provenance}


def _model_reply(record, raw, action_mode):
    if action_mode == 'legal_move':
        return raw, {'rule_legal_constraints': True, 'explanation_generated': False}
    try:
        analysis = parse_explanation(raw)
        move = analysis.get('move')
        proof = validate_explanation(record['fen'], analysis, require_branches=True)
        if proof['valid']:
            try:
                continuation_positions(record, analysis)
            except ValueError:
                proof['valid'] = False
                proof['errors'].append('invalid_history_continuation')
    except (ValueError, TypeError, KeyError):
        move, proof = None, {'valid': False, 'errors': ['invalid_json']}
    return move, proof


def _state(context, moves, turns, max_plies, *, forfeit=False):
    if forfeit:
        return {'status': 'completed', 'winner': 'black' if context['model_color'] == 'red' else 'red',
                'reason': 'raw_model_invalid_move_forfeit', 'moves': moves, 'turns': turns, **context}
    outcome = adjudicate(context['initial_fen'], moves)
    if outcome['ended']:
        return {'status': 'completed', **outcome, 'moves': moves, 'turns': turns, **context}
    if len(turns) == max_plies:
        return {'status': 'censored', 'winner': None, 'reason': 'ply_limit', 'moves': moves,
                'turns': turns, **context}
    return {'status': 'playing', 'moves': moves, 'turns': turns, **context}


def restore_game(opening, model_color, nodes, max_plies, saved=None, action_mode='explanation', move_beams=4):
    """Rebuild a saved game from raw turns and native history, without either player."""
    if (model_color not in {'red', 'black'} or action_mode not in {'explanation', 'legal_move'} or
            type(nodes) is not int or nodes <= 0 or type(max_plies) is not int or max_plies < 0 or
            type(move_beams) is not int or move_beams < 1):
        raise ValueError('Require a known model color/action mode and valid match budgets')
    context = _context(opening, model_color, nodes)
    moves = list(context['opening_moves'])
    history = replay(context['initial_fen'], moves)
    turns = []
    current = _state(context, moves, turns, max_plies)
    if saved is None:
        return current
    if not isinstance(saved, dict) or not isinstance(saved.get('turns'), list):
        raise ValueError('Saved match must contain its raw turn list')
    for turn in saved['turns']:
        if current['status'] != 'playing':
            raise ValueError('Saved match crosses a terminal position or its ply limit')
        fen = history[-1]
        mover = 'red' if fen.split()[1] == 'w' else 'black'
        if (not isinstance(turn, dict) or type(turn.get('ply')) is not int or
                turn['ply'] != len(moves) or turn.get('mover') != mover):
            raise ValueError('Saved turn differs from its complete native history')
        if mover == model_color:
            record = {'initial_fen': context['initial_fen'], 'moves': list(moves), 'history': list(history),
                      'fen': fen, 'question': EXPLANATION_QUESTION}
            move, proof = _model_reply(record, turn.get('raw'), action_mode)
            expected = {'ply': len(moves), 'mover': mover, 'move': move, 'raw': turn.get('raw'),
                        'verification': proof, 'model_oracle_used': False, 'action_mode': action_mode}
            if _game_digest(turn) != _game_digest(expected) or turn.get('model_oracle_used') is not False:
                raise ValueError('Saved model turn differs from its raw answer or native verification')
        else:
            oracle = turn.get('oracle')
            if (not isinstance(oracle, dict) or set(turn) != {'ply', 'mover', 'move', 'oracle'} or
                    turn.get('move') != oracle.get('best_move') or
                    'requested_nodes' in oracle and
                    (type(oracle['requested_nodes']) is not int or oracle['requested_nodes'] != nodes)):
                raise ValueError('Saved opponent turn differs from its requested node budget')
            move = turn['move']
            if not isinstance(move, str) or move not in legal_moves(fen):
                raise ValueError('Saved opponent turn is not a native legal move')
        turns.append(deepcopy(turn))
        forfeit = mover == model_color and (not isinstance(move, str) or move not in legal_moves(fen))
        if not forfeit:
            moves.append(move)
            history.append(play(fen, move))
        current = _state(context, list(moves), list(turns), max_plies, forfeit=forfeit)
    if _game_digest(saved) != _game_digest(current):
        raise ValueError('Saved match result, provenance or moves differ from its raw native replay')
    return current


def play_game(predictor, oracle, opening, model_color, nodes, max_plies, action_mode='explanation', move_beams=4,
              *, resume_state=None, on_progress=None):
    current = restore_game(opening, model_color, nodes, max_plies, resume_state, action_mode, move_beams)
    context = _context(opening, model_color, nodes)
    moves, turns = list(current['moves']), deepcopy(current['turns'])
    history = replay(context['initial_fen'], moves)
    while current['status'] == 'playing':
        fen = history[-1]
        mover = 'red' if fen.split()[1] == 'w' else 'black'
        if mover == model_color:
            record = {'initial_fen': context['initial_fen'], 'moves': list(moves), 'history': list(history), 'fen': fen,
                      'question': EXPLANATION_QUESTION}
            if action_mode == 'legal_move':
                raw = predictor.generate_moves([record], beams=move_beams)[0]
            else:
                raw = predictor.generate(record, max_new_tokens=768)
            move, proof = _model_reply(record, raw, action_mode)
            turns.append({'ply': len(moves), 'mover': mover, 'move': move, 'raw': raw, 'verification': proof,
                          'model_oracle_used': False, 'action_mode': action_mode})
        else:
            result = oracle.choose_move(fen, nodes, context['initial_fen'], moves)
            move = result['best_move']
            turns.append({'ply': len(moves), 'mover': mover, 'move': move, 'oracle': result})
        forfeit = mover == model_color and (not isinstance(move, str) or move not in legal_moves(fen))
        if not forfeit:
            moves.append(move)
            history.append(play(fen, move))
        current = _state(context, list(moves), deepcopy(turns), max_plies, forfeit=forfeit)
        if on_progress is not None:
            on_progress(deepcopy(current))
    return current


def _checkpoint_spec(opening, color, nodes, max_plies, action_mode, move_beams):
    return {'context': _context(opening, color, nodes), 'max_plies': max_plies,
            'action_mode': action_mode, 'move_beams': move_beams}


def _game_digest(game):
    return hashlib.sha256(json.dumps(game, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save_game_checkpoint(path, spec, game):
    atomic_json(path, {'schema_version': 1, 'spec': spec, 'game': game, 'game_sha256': _game_digest(game)})


def load_game_checkpoint(path, opening, color, nodes, max_plies, action_mode, move_beams):
    value = json.loads(Path(path).read_text())
    expected = _checkpoint_spec(opening, color, nodes, max_plies, action_mode, move_beams)
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'spec', 'game', 'game_sha256'} or
            type(value['schema_version']) is not int or value['schema_version'] != 1 or
            _game_digest(value['spec']) != _game_digest(expected) or _game_digest(value['game']) != value['game_sha256']):
        raise ValueError('Saved match checkpoint hash or evaluation specification changed')
    return restore_game(opening, color, nodes, max_plies, value['game'], action_mode, move_beams)


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


def _saved_games(dest, schedule, max_plies, action_mode, move_beams):
    expected_names = {f'game-{i:03d}.json' for i in range(1, len(schedule) + 1)}
    if any(p.name not in expected_names for folder in [dest, dest / 'progress']
           for p in folder.glob('game-*.json')):
        raise ValueError('Saved match files differ from the planned game schedule')
    result, missing, active = {}, False, False
    for i, (opening, color, nodes) in enumerate(schedule, 1):
        checkpoint, completed = dest / 'progress' / f'game-{i:03d}.json', dest / f'game-{i:03d}.json'
        if not checkpoint.exists():
            if completed.exists():
                raise ValueError('Saved game has no bound turn checkpoint')
            missing = True
            continue
        if missing or active:
            raise ValueError('Saved games must form a prefix with at most one active final game')
        game = load_game_checkpoint(checkpoint, opening, color, nodes, max_plies, action_mode, move_beams)
        active = game['status'] == 'playing'
        if completed.exists():
            published = json.loads(completed.read_text())
            if active or _game_digest(published) != _game_digest(game):
                raise ValueError('Saved published game differs from its verified raw turns')
        result[i] = game
    return result


def _checked_completed_run(dest, config, code, inputs, games):
    proof = json.loads((dest / 'manifest.json').read_text())
    required_outputs = {str(dest / name) for name in ['games.jsonl', 'metrics.json', 'contract.json']}
    required_outputs.update(str(dest / f'game-{i:03d}.json') for i in range(1, len(games) + 1))
    required_outputs.update(str(dest / 'progress' / f'game-{i:03d}.json') for i in range(1, len(games) + 1))
    if (proof.get('status') != 'complete' or proof.get('kind') != 'raw_model_full_matches' or
            proof.get('config') != config or proof.get('code') != code or proof.get('inputs') != inputs or
            set(proof.get('outputs', {})) != required_outputs):
        raise ValueError('Completed match run differs from its original evaluation contract')
    bound = BoundInputs()
    for path, identity in proof['outputs'].items():
        bound.bind(path, identity)
    actual = [json.loads(line) for line in (dest / 'games.jsonl').read_text().splitlines() if line.strip()]
    metrics = bound.json(dest / 'metrics.json')
    if (_game_digest(actual) != _game_digest(games) or metrics != proof['verification'] or
            any(metrics.get(k) != v for k, v in match_summary(games).items())):
        raise ValueError('Completed match metrics differ from verified raw games')
    bound.unchanged()
    return metrics


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
    parser.add_argument('--resume', action='store_true', help='Continue a run using its unchanged inputs and source')
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists() and not args.resume:
        raise FileExistsError('Use a fresh match evaluation output')
    if args.resume and not dest.is_dir():
        raise FileNotFoundError('Resume requires an existing match evaluation directory')
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
    config = {k: v for k, v in vars(args).items() if k != 'resume'}
    contract = {'schema_version': 1, 'config': config, 'code': code, 'inputs': core.artifacts}
    schedule = [(opening, color, nodes) for nodes in args.nodes for opening in openings for color in ['red', 'black']]
    if args.resume:
        if (not (dest / 'contract.json').is_file() or
                _game_digest(json.loads((dest / 'contract.json').read_text())) != _game_digest(contract)):
            raise ValueError('Match resume requires its original input, source and evaluation contract')
    else:
        dest.mkdir(parents=True)
        atomic_json(dest / 'contract.json', contract)
    saved = _saved_games(dest, schedule, args.max_plies, args.action_mode, args.move_beams)
    if (dest / 'manifest.json').exists():
        if len(saved) != len(schedule) or any(g['status'] == 'playing' for g in saved.values()):
            raise ValueError('Completed match run has incomplete saved games')
        proof = _checked_completed_run(dest, config, code, core.artifacts, list(saved.values()))
        core.unchanged()
        if code_identity() != code:
            raise ValueError('Match execution source changed; preserve completed outputs')
        print(json.dumps(proof), flush=True)
        return
    started, games = time.monotonic(), []
    predictor, oracle = None, None
    try:
        for i, (opening, color, nodes) in enumerate(schedule, 1):
            game = saved.get(i)
            checkpoint = dest / 'progress' / f'game-{i:03d}.json'
            spec = _checkpoint_spec(opening, color, nodes, args.max_plies, args.action_mode, args.move_beams)
            if game is None or game['status'] == 'playing':
                if predictor is None:
                    from .inference import Predictor
                    predictor = Predictor(args.checkpoint, args.expert_weights)
                    oracle = Pikafish(args.executable, args.weights, threads=1, multipv=1)
                game = play_game(predictor, oracle, opening, color, nodes, args.max_plies,
                                 args.action_mode, args.move_beams, resume_state=game,
                                 on_progress=lambda state: save_game_checkpoint(checkpoint, spec, state))
                save_game_checkpoint(checkpoint, spec, game)
            games.append(game)
            completed = dest / f'game-{i:03d}.json'
            if not completed.exists():
                atomic_json(completed, game)
            print(json.dumps(match_summary(games)), flush=True)
    finally:
        if oracle is not None:
            oracle.close()
    write_jsonl(dest/'games.jsonl', games)
    proof = {**match_summary(games), 'action_mode': args.action_mode,
             'openings': len(openings), 'opening_split': opening_proof['config']['split'] if opening_proof else None,
             'both_model_colors_at_each_opening_and_budget': True,
             'rule_legal_constraints': args.action_mode == 'legal_move',
             'explanation_quality_evaluated': False,
             'by_opponent_nodes': {str(n): match_summary([g for g in games if g['opponent_nodes'] == n])
                                                         for n in args.nodes},
             'seconds': time.monotonic()-started, 'elapsed_scope': 'last_invocation',
             'completed_games_reused': sum(g['status'] != 'playing' for g in saved.values()),
             'saved_turns_reused': sum(len(g['turns']) for g in saved.values()),
             'rule_profile': 'pyffish-0.0.90-xiangqi-AXF'}
    atomic_json(dest/'metrics.json', proof)
    core.unchanged()
    if code_identity() != code:
        raise ValueError('Match execution source changed; preserve partial outputs')
    completed = manifest('raw_model_full_matches', config, (),
                         [dest/'games.jsonl', dest/'metrics.json', dest/'contract.json',
                          *sorted(dest.glob('game-*.json')), *sorted((dest/'progress').glob('game-*.json'))],
                         proof, code=code)
    completed['inputs'] = core.artifacts
    core.unchanged();atomic_json(dest/'manifest.json', completed)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
