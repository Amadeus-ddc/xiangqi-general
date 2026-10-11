import copy
import json
import sys
import pytest

from xqgeneral.evidence import atomic_json, history_key, position_key
from xqgeneral.rules import START_FEN, replay
from xqgeneral.search_distillation import SearchMiner, descend, root_evaluation
from xqgeneral.search_critical import (checked_acceptance_policy, comparison_boundary, decide,
                                     finish_critical, first_critical_divergence, parse_critical, parsed_sequence)
from test_search_distillation import ControlledOracle, ControlledPredictor, root_record
from test_candidate_fill_and_reuse import POLICY


class Predictor(ControlledPredictor):
    def __init__(self, original_pv=None, illegal_suffix=False, omit_recommendation=False):
        super().__init__()
        self.original_pv, self.illegal_suffix, self.omit_recommendation = original_pv, illegal_suffix, omit_recommendation
        self.raws = []

    def generate(self, row, question, budget, **options):
        value=json.loads(super().generate(row,question,budget))
        if not row['moves'] and self.original_pv is not None:
            value['pv']=self.original_pv
        if row['moves'] and self.illegal_suffix:
            value['pv'].append('a0a0')
        if row['moves'] and self.omit_recommendation:
            value.pop('move');value.pop('candidates')
        raw=json.dumps(value,ensure_ascii=False);self.raws.append(raw);return raw


def miner(predictor=None, oracle=None, **options):
    return SearchMiner(predictor or Predictor(),oracle or ControlledOracle(),options.pop('reserved',set()),
        generation_policy=POLICY,max_root_retries=3,root_candidate_policy='paper_fill',child_contract='move_eval',
        acceptance_policy='paper_recursive',reuse_child_analysis=options.pop('reuse_child_analysis',True),**options)


@pytest.mark.parametrize('old_line',[['h0g2','b9c7'],['h0g2'],['h0g2','b9c7','e3e4']])
def test_equal_or_prefix_equivalent_lines_can_improve_numerical_evaluation(old_line):
    query,trace,reason=miner(Predictor(original_pv=old_line)).mine(root_record())
    assert reason=='accepted_for_consolidation' and query['target_fields']['move']=='h0g2'
    decision=trace[0]['paper_decision'];divergence=trace[0]['paper_divergence']
    assert decision['action']=='accept' and divergence['kind']=='no_divergence'
    assert divergence['new_eval_error']<divergence['old_eval_error']
    assert json.loads(trace[0]['root_generation']['raw'])['pv']==old_line


def test_illegal_original_line_is_repaired_without_rewriting_raw_generation():
    query,trace,reason=miner(Predictor(original_pv=['a0a0'])).mine(root_record())
    assert query is not None and reason=='accepted_for_consolidation'
    assert trace[0]['paper_divergence']['reason']=='new line repairs illegal first divergence'
    assert trace[0]['root_generation']['analysis']['pv']==['a0a0']
    assert trace[0]['root_generation']['verification']['valid'] is False


def test_child_illegal_suffix_retains_audited_prefix_and_raw_failures():
    predictor=Predictor(illegal_suffix=True)
    query,trace,reason=miner(predictor).mine(root_record())
    assert query is not None and reason=='accepted_for_consolidation'
    assert query['target_fields']['pv']==['h0g2','b9c7']
    for child in trace[0]['children']:
        assert child['analysis']['pv']==['b9c7','a0a0']
        assert child['raw'] in predictor.raws and child['verification']['valid'] is False
        assert child['sanitation']['accepted_uci']==['b9c7']
        assert child['sanitation']['illegal_at']['ply']==2
    content=json.loads(query['messages'][1]['content'])
    assert all(c['used_pv']==['b9c7'] for c in content['child_analyses'])
    assert all(c['full_output_verification']['valid'] is False for c in content['child_analyses'])


def test_numerical_and_clean_prefix_acceptance_does_not_require_a_child_recommendation():
    query,trace,reason=miner(Predictor(omit_recommendation=True)).mine(root_record())
    assert query is not None and reason=='accepted_for_consolidation'
    assert all(c['child_acceptance']['valid'] is False for c in trace[0]['children'])
    assert all(c['numerical_acceptance']['valid'] for c in trace[0]['children'])


class DeepOracle(ControlledOracle):
    def analyze(self,fen,nodes,initial_fen,moves,searchmoves=None):
        if len(moves)<2:
            return super().analyze(fen,nodes,initial_fen,moves,searchmoves)
        scores={'e3e4':400,'b0c2':-400}
        chosen=searchmoves or ['e3e4','b0c2']
        return {'best_move':'e3e4' if 'e3e4' in chosen else chosen[0],
            'candidates':[{'move':m,'score_type':'cp','score':scores[m],'pv':[m],'wdl':None} for m in chosen]}


def test_truncation_boundary_prevents_a_removed_mistake_hiding_behind_a_better_evaluation():
    class LongPredictor(Predictor):
        def generate(self,row,question,budget,**options):
            value=json.loads(super().generate(row,question,budget,**options))
            if row['moves']:value['pv'].append('b0c2')
            return json.dumps(value)
    query,trace,reason=miner(LongPredictor(original_pv=['h0g2','b9c7','e3e4']),DeepOracle(),max_depth=0).mine(root_record())
    assert query is None and reason=='recursion_limit'
    selected=next(c for c in trace[0]['children'] if c['move']=='h0g2')
    assert selected['sanitation']['accepted_uci']==['b9c7']
    divergence=trace[0]['paper_divergence']
    assert divergence['used_truncation_boundary'] and divergence['kind']=='divergence'
    assert divergence['old']['uci']=='e3e4' and divergence['new']['uci']=='b0c2'
    assert not divergence['accept'] and divergence['new_score']<divergence['old_score']
    assert trace[0]['paper_decision']['recurse_record_id']==selected['record']['id']


def child(label,truth=.8,predicted=.8,order=0,valid=True):
    return {'move':label,'record':{'id':label,'feature_key':label},'true_root_probability':truth,
            'predicted_root_probability':predicted,'predicted_root_order':(0,order),
            'sanitation':{'valid_checked_prefix':valid}}


@pytest.mark.parametrize('cause,expected,reason',[
    ('malformed','a','child evaluation remained malformed'),
    ('ordering','b','incorrect inaccurate child ordering'),
    ('ordering_other','a','incorrect inaccurate child ordering'),
    ('truth','a','inferred evaluation outside truth limit'),
    ('prefix','c','child exhausted clean-prefix attempts'),
    ('divergence','a','controlled no improvement'),
])
def test_mandatory_recursion_priority_and_offending_child(cause,expected,reason):
    children=[child('a',order=30),child('b',truth=.6,predicted=.6,order=20),child('c',truth=.5,predicted=.5,order=10)]
    divergence={'accept':True,'reason':'controlled improvement'}
    if cause=='malformed':children[0].pop('predicted_root_probability')
    if cause in {'ordering','ordering_other'}:
        children[1].update(predicted_root_probability=.9,predicted_root_order=(0,40))
        if cause=='ordering_other':children[0]['predicted_root_probability']=.1
    if cause=='truth':children[0]['predicted_root_probability']=.95
    if cause=='prefix':children[2]['sanitation']['valid_checked_prefix']=False
    if cause=='divergence':divergence={'accept':False,'reason':'controlled no improvement'}
    original=copy.deepcopy(children)
    decision=decide(children,.2,.8,divergence)
    assert decision['action']=='recurse' and decision['recurse_record_id']==expected
    assert decision['reason']==reason and children==original


def test_eval_truth_tolerance_includes_exact_threshold():
    children=[child('a',truth=0,predicted=.1,order=1)]
    assert decide(children,.5,0,{'accept':True,'reason':'improved'})['action']=='recurse'


def test_terminal_histories_need_no_clean_move_and_do_not_extend_the_game():
    moves=['b0c2','b9c7','c2b0','c7b9']*2;history=replay(START_FEN,moves)
    row=dict(root_record(),moves=moves,history=history,fen=history[-1],feature_key=history_key(history))
    parsed=parse_critical(row,['b0c2']);sanitation=finish_critical(parsed,{})
    assert parsed['terminal'] and parsed['legal_steps']==[]
    assert sanitation['valid_checked_prefix'] and sanitation['accepted_ply']==0


@pytest.mark.parametrize('drop,accepted',[(.099,2),(.10,1),(.101,1)])
def test_mistake_threshold_and_removed_boundary(drop,accepted):
    parsed=parse_critical(root_record(),['b0c2','b9c7'])
    audits={(p['record']['feature_key'],p['uci']):{'winrate_drop':0 if p['ply']==1 else drop} for p in parsed['legal_steps']}
    sanitation=finish_critical(parsed,audits)
    assert sanitation['accepted_ply']==accepted
    if accepted==1:assert comparison_boundary(sanitation)['uci']=='b9c7'


def test_illegal_sequences_preserve_first_divergence_and_complete_past():
    root=root_record();old=parsed_sequence(parse_critical(root,['b0c2','a0a0']))
    new=parsed_sequence(parse_critical(root,['b0c2','b9c7']))
    divergence=first_critical_divergence(root,old,new)
    assert divergence['kind']=='divergence' and divergence['record']['moves']==['b0c2']
    assert divergence['old']['legal'] is False and divergence['new']['legal'] is True
    assert first_critical_divergence(root,old,old)['kind']=='identical_illegal'


def test_raw_unused_branch_reservation_is_not_hidden_by_prefix_sanitation():
    class BranchPredictor(Predictor):
        def generate(self,row,question,budget,**options):
            value=json.loads(super().generate(row,question,budget,**options))
            if row['moves']==['h0g2']:value['branches']=[{'pv':['h9g7']}]
            return json.dumps(value)
    reserved={position_key(replay(START_FEN,['h0g2','h9g7'])[-1])}
    assert miner(BranchPredictor(),reserved=reserved).mine(root_record())[2]=='reserved_child_continuation'


@pytest.mark.parametrize('policy,candidates,contract,retries,clean',[
    ('unknown','paper_fill','move_eval',3,1),('paper_recursive','strict','move_eval',3,1),
    ('paper_recursive','paper_fill','full',3,1),('paper_recursive','paper_fill','move_eval',None,1),
    ('paper_recursive','paper_fill','move_eval',3,False),('paper_recursive','paper_fill','move_eval',3,0),
    ('strict','strict','full',None,2),
])
def test_invalid_profiles_are_rejected_before_generation(policy,candidates,contract,retries,clean):
    with pytest.raises(ValueError):checked_acceptance_policy(policy,candidates,contract,retries,clean)


def test_child_mate_zero_is_an_immediate_win_from_the_parent():
    assert root_evaluation({'evaluation':{'type':'mate','value':0,'perspective':'side_to_move'}})['value']==1


def test_selected_close_alternative_heads_the_target_without_changing_native_rank_trace():
    class CloseOracle:
        def analyze(self,fen,nodes,initial_fen,moves,searchmoves=None):
            roots={'h0g2':200,'b0c2':190,'c3c4':-300}
            scores=roots if not moves else {'b9c7':-roots[moves[0]]}
            chosen=searchmoves or list(scores)
            return {'best_move':max(chosen,key=scores.get),'candidates':[
                {'move':m,'score_type':'cp','score':scores[m],'pv':[m],'wdl':None} for m in chosen]}
    class ClosePredictor(Predictor):
        def generate(self,row,question,budget,**options):
            value=json.loads(super().generate(row,question,budget,**options))
            if not row['moves']:
                value.update(move='c3c4',pv=['c3c4','b9c7']);value['evaluation']['value']=-300
            elif row['moves']==['b0c2']:value['evaluation']['value']=-210
            return json.dumps(value)
    query,trace,reason=miner(ClosePredictor(),CloseOracle()).mine(root_record())
    assert query is not None and reason=='accepted_for_consolidation'
    assert query['target_fields']['move']==query['target_fields']['candidates'][0]=='b0c2'
    assert trace[0]['children'][0]['move']=='h0g2'
    assert query['selected_root_move_loss']<.05


def test_rejected_equal_line_recurses_and_reuses_the_original_numerical_child():
    from test_candidate_fill_and_reuse import TreeOracle,TreePredictor
    class EqualPredictor(TreePredictor):
        def generate(self,row,question,budget,**options):
            value=json.loads(super().generate(row,question,budget,**options))
            if not row['moves']:
                value['pv']=['h0g2','b9c7'];value['evaluation']['value']=200
            return json.dumps(value)
    predictor=EqualPredictor();search=miner(predictor,TreeOracle(),max_depth=1)
    query,trace,reason=search.mine(root_record())
    assert query is None and reason=='recursion_limit' and len(trace)==2
    child=next(c for c in trace[0]['children'] if c['move']=='h0g2')
    reused=trace[1]['root_generation']
    assert trace[0]['paper_decision']['reason']=='prefix-equivalent line does not improve evaluation'
    assert child['raw']==reused['raw'] and child['generation']==reused['generation']
    assert reused['root_evaluation_acceptance']['attempt']==0
    assert predictor.calls.count('r-h0g2')==1
    uncached=EqualPredictor();miner(uncached,TreeOracle(),max_depth=1,reuse_child_analysis=False).mine(root_record())
    assert uncached.calls.count('r-h0g2')==2


@pytest.mark.parametrize('changed_flag,changed_value',[('--acceptance-policy','strict'),('--min-clean-ply','2')])
def test_recursive_profile_flows_through_real_cli_and_resume_rejects_changes(tmp_path,monkeypatch,changed_flag,changed_value):
    from test_search_pilot import ControlledChain,setup_pilot
    from xqgeneral import search_pilot,search_distillation
    pipeline,validation,config,query,_=setup_pilot(tmp_path,monkeypatch)
    policy=tmp_path/'sampling.json';atomic_json(policy,POLICY)
    settings=json.loads(config.read_text());settings.update(student_generation_config=str(policy),max_root_retries=3,
        root_candidate_policy='paper_fill',reuse_child_analysis=True,acceptance_policy='paper_recursive',min_clean_ply=1)
    atomic_json(config,settings);chain=ControlledChain(monkeypatch,query);output=tmp_path/'pilot'
    search_pilot.run_pilot(pipeline,validation,'owned-validation',config,output)
    command=next(c for c in chain.commands if 'xqgeneral.search_distillation' in c)
    assert command[command.index('--acceptance-policy')+1]=='paper_recursive'
    assert command[command.index('--min-clean-ply')+1]=='1'
    proof=json.loads((output/'mining/manifest.json').read_text())
    assert proof['verification']['acceptance_policy']=='paper_recursive'
    (output/'mining/manifest.json').unlink()
    args=command[command.index('xqgeneral.search_distillation'):]
    args[args.index(changed_flag)+1]=changed_value;monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError,match='continuation inputs'):search_distillation.main()
    assert chain.student_loads==1
