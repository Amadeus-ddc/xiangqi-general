from xqgeneral.evaluate_explanations import judge_move
from xqgeneral.evaluate_moves import move_summary, parsed_move
from xqgeneral.rules import START_FEN, replay


def test_raw_move_quality_requires_legal_nonterminal_move_without_repairs():
    root = {'fen':START_FEN,'initial_fen':START_FEN,'moves':[],
            'answer':'b0c2','teacher':{'neural_prose_generated':False}}
    assert parsed_move(' b0c2\n') == 'b0c2'
    assert parsed_move('建议 b0c2') is None
    invalid = judge_move(None,root,'a0a0',100)
    assert not invalid['first_move_legal'] and not invalid['first_move_no_mistake']
    assert 'first_move_expected_score_loss' not in invalid
    moves = ['b0c2','b9c7','c2b0','c7b9']*2
    terminal = dict(root,moves=moves,fen=replay(START_FEN,moves)[-1])
    assert 'terminal_root' in judge_move(None,terminal,'b0c2',100)

    class RestrictedOracle:
        def __init__(self):self.calls=[]
        def analyze(self,fen,nodes,initial_fen,moves,searchmoves=None):
            self.calls.append(searchmoves)
            move = searchmoves[0] if searchmoves else 'b0c2'
            return {'best_move':move,'candidates':[{'move':move,'wdl':[0,0,1000] if searchmoves else [1000,0,0]}]}
    oracle = RestrictedOracle()
    bad = judge_move(oracle,root,'h0g2',100)
    assert oracle.calls == [None,['h0g2']]
    assert bad['first_move_legal'] and not bad['first_move_no_mistake']
    assert bad['first_move_expected_score_loss'] == 1
    summary = move_summary([{'record':root,'move':None,'judgment':invalid},
                            {'record':root,'move':'h0g2','judgment':bad}])
    assert summary['legal_rate'] == .5 and summary['no_mistake_rate'] == 0
    assert summary['mean_expected_score_loss_conditional'] == 1 and summary['legal_scored_moves'] == 1
    assert summary['oracle_repairs'] == 0 and summary['teacher_best_move_exact_rate'] == 0
