"""Pikafish's pinned material-conditioned WDL model, including UCI cp scaling.

Coefficients/formulas are derived from vendor/pikafish/src/uci.cpp (GPLv3).
This is the engine's fitted score model, not a measured human winning chance.
"""
import math
from .rules import piece_map

MATERIAL = {'r': 10, 'h': 5, 'c': 5, 'e': 3, 'a': 2, 'p': 1, 'k': 0}
AS = (220.59891365, -810.35730430, 928.68185198, 79.83955423)
BS = (61.99287416, -233.72674182, 325.85508322, -68.72720854)


def evaluation_probability(fen, evaluation):
    if (evaluation.get('perspective') != 'side_to_move' or type(evaluation.get('value')) is not int or
            evaluation.get('type') not in {'cp', 'mate'}):
        raise ValueError('Invalid evaluation contract')
    value = evaluation['value']
    if evaluation['type'] == 'mate':
        return 1.0 if value > 0 else 0.0
    material = sum(MATERIAL[p.lower()] for p in piece_map(fen).values())
    m = max(17, min(110, material)) / 65.0
    a = ((AS[0] * m + AS[1]) * m + AS[2]) * m + AS[3]
    b = ((BS[0] * m + BS[1]) * m + BS[2]) * m + BS[3]
    internal = value * a / 100.0
    def won(score):
        exponent = max(-80.0, min(80.0, (a - score) / b))
        return int(0.5 + 1000 / (1 + math.exp(exponent)))
    win, loss = won(internal), won(-internal)
    return (win + (1000 - win - loss) / 2) / 1000.0


def candidate_probability(fen, candidate):
    wdl = candidate.get('wdl')
    if wdl is not None:
        if len(wdl) != 3 or any(v < 0 for v in wdl) or sum(wdl) != 1000:
            raise ValueError('Unexpected engine WDL units')
        return (wdl[0] + wdl[1] / 2) / 1000.0
    return evaluation_probability(fen, {'type': candidate['score_type'], 'value': candidate['score'],
                                       'perspective': 'side_to_move'})
