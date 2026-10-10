"""Native target classes and complete terminal-source footprint semantics."""
from .evidence import position_key
from .rules import gives_check, in_check, legal_moves, replay
from .symmetry import mirror_fen

NEAR_MATE_PLIES = 6


def native_classes(fen, mate_distance=None):
    """Classes describe the queried board; near-mate needs an actual source witness."""
    moves, check = legal_moves(fen), in_check(fen)
    classes = ['generic']
    if any(gives_check(fen, move) for move in moves):
        classes.append('have_check')
    if check:
        classes.append('in_check')
    if not moves:
        classes.append('mate' if check else 'stalemate')
    if check and mate_distance is not None and 1 <= mate_distance <= NEAR_MATE_PLIES:
        classes.append('near_mate_check')
    return classes



def root_footprint(root):
    """Reserve the maximum future, the entire actual mate witness, and both colors."""
    line = (root.get('paper_generated_terminal_witness') or
            root['paper_recorded_mate_witness'] or root['future_moves'])
    fens = replay(root['fen'], line)
    if root.get('extension_is_recorded_source_move') is False:
        # A generated one-ply descendant retains its supplied parent board.
        fens += root['history']
    return {position_key(fen) for fen in fens} | {position_key(mirror_fen(fen)) for fen in fens}
