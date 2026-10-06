import pytest
from xqgeneral.evidence import history_key,position_key,write_jsonl
from xqgeneral.policy_data import descendant_contexts,move_label,extra_training_contexts
from xqgeneral.rules import START_FEN,replay


def test_policy_descendants_keep_train_ownership_and_exclude_heldout_boards():
    moves=['b0c2'];history=replay(START_FEN,moves)
    row={'id':'root','feature_key':history_key(history),'initial_fen':START_FEN,'moves':moves,
         'history':history,'fen':history[-1],'split':'train','game_id':'train-game','provenance':'engine',
         'query':{},'future_moves':[]}
    query={'id':'root-query','feature_key':row['feature_key'],'record':row,
           'oracle':{'best_move':'b9c7','candidates':[{'pv':['b9c7','h0g2','h9g7']}]}}
    forbidden={position_key(replay(START_FEN,['b0c2','b9c7','h0g2'])[-1])}
    descendants=descendant_contexts([query],forbidden,8,1)
    assert len(descendants)==2
    assert all(r['split']=='train' and r['game_id']=='train-game' for r in descendants)
    assert all(position_key(r['fen']) not in forbidden and r['history'][-1]==r['fen'] for r in descendants)
    label=move_label(query)
    assert label['answer']=='b9c7' and label['future_moves']==['b9c7']
    assert not label['teacher']['neural_prose_generated']
    validation=dict(query,record=dict(row,split='validation'))
    assert descendant_contexts([validation],set(),8,1)==[]


def test_extra_policy_contexts_verify_ownership_history_and_future_exclusion(tmp_path):
    moves=['b0c2'];history=replay(START_FEN,moves)
    row={'id':'new','feature_key':history_key(history),'initial_fen':START_FEN,'moves':moves,
         'history':history,'fen':history[-1],'split':'train','game_id':'new-training-game',
         'provenance':'engine-selfplay','future_moves':['b9c7']}
    path=tmp_path/'extra.jsonl';write_jsonl(path,[row,row])
    selected,rejected=extra_training_contexts([path],[],set(),set())
    assert selected==[row] and rejected=={'duplicate_history':1}
    heldout={position_key(replay(START_FEN,['b0c2','b9c7'])[-1])}
    assert extra_training_contexts([path],[],heldout,set())==([],{'heldout_root_or_future':2})
    with pytest.raises(ValueError,match='training games'):
        extra_training_contexts([path],[],set(),{'new-training-game'})
    write_jsonl(path,[dict(row,feature_key='incorrect')])
    with pytest.raises(ValueError,match='full history'):
        extra_training_contexts([path],[],set(),set())
    write_jsonl(path,[dict(row,split='validation')])
    with pytest.raises(ValueError,match='training games'):
        extra_training_contexts([path],[],set(),set())
