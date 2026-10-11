import copy
import json
import sys

import pytest

from xqgeneral.evidence import atomic_json, history_key
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, legal_moves, replay
from xqgeneral.search_distillation import SearchMiner, descend
from test_search_distillation import ControlledOracle, ControlledPredictor, root_record

POLICY = {'do_sample': True, 'temperature': .6, 'top_k': 20, 'top_p': .95, 'max_new_tokens': 8}


def miner(predictor=None, oracle=None, **kwargs):
    return SearchMiner(predictor, oracle or ControlledOracle(), set(), generation_policy=POLICY,
        generation_seed=20260828, max_root_retries=3, **kwargs)


@pytest.mark.parametrize('candidates', [[], ['b0c2'], ['a0a0', 'b0c2', 'b0c2'], None, 'bad', [None, {'bad': 1}]])
def test_fill_keeps_the_raw_student_output_and_records_every_added_candidate(candidates):
    class Predictor(ControlledPredictor):
        def generate(self, row, question, budget, **options):
            value = json.loads(super().generate(row, question, budget))
            if not row['moves']: value['candidates'] = candidates
            return json.dumps(value, ensure_ascii=False)
    query, trace, reason = miner(Predictor(), root_candidate_policy='paper_fill').mine(root_record())
    assert query is not None and reason == 'accepted_for_consolidation'
    raw = json.loads(trace[0]['root_generation']['raw'])
    assert raw['candidates'] == candidates
    assert trace[0]['root_generation']['analysis'] == raw
    preparation = trace[0]['candidate_preparation']
    assert preparation['legal_model_candidates'] == ['b0c2']
    assert preparation['engine_filled_candidates'] == ['h0g2', 'c3c4']
    assert preparation['final_candidates'] == ['h0g2', 'b0c2', 'c3c4']
    assert preparation['sources']['b0c2'] == 'model_preferred_move'
    assert preparation['sources']['h0g2'] == 'engine_multipv_fill'
    assert len(query['target_fields']['candidates']) == 3


def test_preferred_candidates_and_pv_are_filtered_in_model_order_before_fill():
    row = root_record();search = miner(root_candidate_policy='paper_fill')
    value = {'move': 'a0a0', 'candidates': ['b0c2', 'b0c2', None, 'h0g2'],
             'pv': ['c3c4', 'b9c7', 'e3e4']}
    original = copy.deepcopy(value)
    candidates, proof = search.prepare_candidates(row, value, ControlledOracle().analyze(row['fen'], 1, row['initial_fen'], []))
    assert candidates == ['b0c2', 'h0g2', 'c3c4'] and value == original
    assert proof['engine_filled_candidates'] == []
    assert proof['sources']['c3c4'] == 'model_original_pv'


def test_empty_model_recommendations_use_only_actual_engine_suggestions():
    row = root_record();search = miner(root_candidate_policy='paper_fill')
    value = {'move': 'a0a0', 'candidates': [], 'pv': []}
    engine = ControlledOracle().analyze(row['fen'], 1, row['initial_fen'], [])
    candidates, proof = search.prepare_candidates(row, value, engine)
    assert candidates == proof['engine_filled_candidates'] == ['h0g2', 'b0c2', 'c3c4']
    assert proof['legal_model_candidates'] == []
    engine['candidates'] = engine['candidates'][:1]
    with pytest.raises(ValueError, match='too few distinct legal'): search.prepare_candidates(row, value, engine)


def test_all_bad_complete_model_candidates_still_inject_the_best_and_rank_before_generation():
    class Predictor(ControlledPredictor):
        def generate(self, row, question, budget, **kwargs):
            return super().generate(row, question, budget)
    query, trace, reason = miner(Predictor(inject=True), root_candidate_policy='paper_fill').mine(root_record())
    assert query is not None and reason == 'accepted_for_consolidation'
    assert trace[0]['injected_oracle_move']
    assert trace[0]['candidate_preparation']['engine_filled_candidates'] == []
    assert trace[0]['candidate_preparation']['final_candidates'][0] == 'h0g2'
    assert trace[0]['candidate_preparation']['sources']['h0g2'] == 'engine_best_injection'


class TreeOracle:
    def rows(self, row):
        moves = row['moves']
        if not moves:
            return [('h0g2', 200), ('b0c2', -200), ('c3c4', -300)]
        if len(moves) == 1:
            score = {'b0c2': 200, 'h0g2': -200, 'c3c4': 300}[moves[0]]
            return [('b9c7', score), ('h9g7', score - 100), ('c6c5', score - 200)]
        parent = dict(row, moves=moves[:-1])
        score = dict(self.rows(parent))[moves[-1]]
        return [(move, -score) for move in legal_moves(row['fen'])[:3]]

    def analyze(self, fen, nodes, initial_fen, moves, searchmoves=None):
        row = {'fen': fen, 'moves': moves}
        scores = self.rows(row)
        if searchmoves: scores = [(m, dict(scores)[m]) for m in searchmoves]
        return {'best_move': scores[0][0], 'candidates': [
            {'move': m, 'score_type': 'cp', 'score': s, 'pv': [m], 'wdl': None} for m,s in scores]}


class TreePredictor:
    def __init__(self, malformed_child=False):
        self.calls, self.raws = [], []
        self.malformed_child = malformed_child

    def generate(self, row, question, budget, **options):
        self.calls.append(row['id'])
        moves = row['moves']
        if not moves:
            move, pv, score, candidates = 'b0c2', ['b0c2', 'b9c7'], -200, ['b0c2','h0g2','c3c4']
        else:
            rows = TreeOracle().rows(row);move,score = rows[0];pv,candidates = [move], [m for m,_ in rows]
        value = {'move': move, 'pv': pv, 'candidates': candidates, 'facts': move_facts(row['fen'], move),
            'evaluation': {'type':'cp','value':score,'perspective':'side_to_move'}, 'explanation':'受控输出，逐条保留原始错误。'}
        if moves == ['b0c2']:
            if self.malformed_child and self.calls.count(row['id']) == 1: value.pop('evaluation')
            elif not self.malformed_child: value['facts']['piece'] = '黑炮'
        raw = json.dumps(value,ensure_ascii=False);self.raws.append(raw);return raw


def test_recursive_root_reuses_child_without_mutating_its_failure_or_consuming_a_new_seed():
    predictor = TreePredictor();search = miner(predictor,TreeOracle(),root_candidate_policy='paper_fill',reuse_child_analysis=True,max_depth=1)
    query,trace,reason = search.mine(root_record())
    assert len(trace) == 2
    source = next(c for c in trace[0]['children'] if c['move'] == 'b0c2')
    reused = trace[1]['root_generation']
    assert source['raw'] == reused['raw'] and source['analysis'] == reused['analysis']
    assert source['generation'] == reused['generation']
    assert source['child_acceptance']['valid'] is False
    assert reused['root_evaluation_acceptance'] == {'valid':True,'errors':[],'attempt':0}
    assert reused['reused_child_analysis']['source_child_acceptance'] == source['child_acceptance']
    assert source['verification']['errors'] == reused['verification']['errors'] == ['incorrect_move_facts']
    assert predictor.calls.count('r-b0c2') == 1 and search.counts['reused_child_root_analyses'] == 1
    other = TreePredictor();uncached = miner(other,TreeOracle(),root_candidate_policy='paper_fill',max_depth=1)
    assert uncached.mine(root_record())[2] == reason
    assert other.calls.count('r-b0c2') == 2


def test_malformed_cached_evaluation_is_retained_as_attempt_zero_then_regenerated():
    predictor = TreePredictor(malformed_child=True)
    search = miner(predictor,TreeOracle(),root_candidate_policy='paper_fill',reuse_child_analysis=True,max_depth=1)
    query,trace,reason = search.mine(root_record())
    attempts = trace[1]['root_generation_attempts']
    assert [a['root_evaluation_acceptance']['attempt'] for a in attempts] == [0,1]
    assert [a['root_evaluation_acceptance']['valid'] for a in attempts] == [False,True]
    assert 'evaluation' not in json.loads(attempts[0]['raw'])
    assert 'evaluation' in json.loads(attempts[1]['raw'])
    assert predictor.calls.count('r-b0c2') == 2
    assert search.counts['unusable_cached_child_evaluations'] == 1 and search.counts['reused_child_root_analyses'] == 0


def test_same_fen_with_different_complete_past_cannot_use_a_cached_analysis():
    moves = ['b0c2','b9c7','c2b0','c7b9','b0c2'];history = replay(START_FEN,moves)
    row = dict(root_record(),moves=moves,history=history,fen=history[-1]);generation = {
        'record': row, 'raw':'unchanged', 'analysis':{}, 'verification':{'valid':False,'errors':[]},
        'child_acceptance':{'valid':False,'errors':[]}}
    other_moves = ['h0g2','h9g7','g2h0','g7h9','b0c2'];other_history = replay(START_FEN,other_moves)
    # Even a caller declaring the same key cannot bypass the actual history binding.
    changed = dict(row,moves=other_moves,history=other_history,fen=other_history[-1])
    assert changed['fen'] == row['fen']
    with pytest.raises(ValueError,match='complete recursive root history'):
        miner(reuse_child_analysis=True).root_analysis(changed,generation)


def test_filled_child_reserved_positions_are_rejected_before_child_generation():
    predictor = TreePredictor();row = root_record();reserved = {' '.join(descend(row,'h0g2')['fen'].split()[:2])}
    search = SearchMiner(predictor,TreeOracle(),reserved,generation_policy=POLICY,max_root_retries=3,
        root_candidate_policy='paper_fill',reuse_child_analysis=True)
    assert search.mine(row)[2] == 'reserved_child'
    assert predictor.calls == ['r']


def test_changed_extensions_are_bound_before_any_resumed_student_load(tmp_path,monkeypatch):
    from test_search_pilot import ControlledChain,setup_pilot
    from xqgeneral import search_pilot,search_distillation
    pipeline,validation,config,query,_ = setup_pilot(tmp_path,monkeypatch)
    policy = tmp_path/'sampling.json';atomic_json(policy,POLICY)
    settings=json.loads(config.read_text());settings.update(student_generation_config=str(policy),max_root_retries=3,
        root_candidate_policy='paper_fill',reuse_child_analysis=True);atomic_json(config,settings)
    chain=ControlledChain(monkeypatch,query);output=tmp_path/'pilot'
    search_pilot.run_pilot(pipeline,validation,'owned-validation',config,output)
    command=next(c for c in chain.commands if 'xqgeneral.search_distillation' in c)
    assert command[command.index('--root-candidate-policy')+1]=='paper_fill' and '--reuse-child-analysis' in command
    proof=json.loads((output/'mining/manifest.json').read_text())
    assert proof['config']['reuse_child_analysis'] is True and proof['verification']['candidate_preparation_is_mining_only']
    (output/'mining/manifest.json').unlink()
    args=command[command.index('xqgeneral.search_distillation'):];args.remove('--reuse-child-analysis')
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError,match='continuation inputs'):search_distillation.main()
    assert chain.student_loads == 1


@pytest.mark.parametrize('kwargs',[{'root_candidate_policy':'unknown'},{'reuse_child_analysis':1},
    {'root_candidate_policy':'paper_fill'},{'reuse_child_analysis':True}])
def test_invalid_or_unbound_extension_policy_rejected(kwargs):
    with pytest.raises(ValueError):SearchMiner(None,None,set(),**kwargs)
