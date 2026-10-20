"""Retain an extra strict score for the same best move in label reference jobs."""
from datetime import datetime, timezone

from .oracle import Pikafish, parse_analysis
from .rules import adjudicate, legal_moves, replay


RECOVERY_POLICY = 'restricted_same_selected_best_after_missing_exact_score'


class ReferencePikafish(Pikafish):
    """Keep incomplete queries; never replace an annotation or selected best move."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.begin_row(None)

    def begin_row(self, index):
        self.row_index = index
        self.recoveries = []
        self.requests = []
        self.last_request = None

    def analyze(self, fen, nodes=1000000, initial_fen=None, moves=(), searchmoves=None):
        request = dict(fen=fen, nodes=nodes, initial_fen=initial_fen, moves=list(moves),
                       searchmoves=searchmoves, raw_output=None)
        self.last_request = request
        output = self.search(fen, nodes, initial_fen, moves, searchmoves)
        request['raw_output'] = output
        self.requests.append(request)
        try:
            return parse_analysis(fen, output, nodes)
        except RuntimeError as error:
            if str(error) != 'Engine completed without a scored best-move candidate' or searchmoves:
                raise
        selected = output[-1].split()[1]
        if (selected not in legal_moves(fen) or initial_fen is None or
                replay(initial_fen, moves)[-1] != fen or adjudicate(initial_fen, moves)['ended']):
            raise ValueError('An additional score requires the same legal, nonterminal full history')
        additional = dict(fen=fen, nodes=nodes, initial_fen=initial_fen, moves=list(moves),
                          searchmoves=[selected], raw_output=None)
        self.last_request = dict(additional, original_unscored_request=request)
        extra = self.search(fen, nodes, initial_fen, moves, searchmoves=[selected])
        additional['raw_output'] = extra
        self.last_request = dict(additional, original_unscored_request=request)
        self.requests.append(additional)
        exact = parse_analysis(fen, extra, nodes)
        if exact['best_move'] != selected:
            raise RuntimeError('The additional score changed the originally selected best move')
        self.recoveries.append(dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
            row_index=self.row_index, policy=RECOVERY_POLICY, original_request=request,
            additional_request=additional, original_selected_move=selected,
            selected_move_unchanged=True, strict_result=exact, additional_engine_requests=1,
            teacher_or_student_prediction_rewritten=False))
        return exact
