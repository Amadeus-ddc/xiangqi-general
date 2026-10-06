from xqgeneral.evidence import history_key,position_key
from xqgeneral.policy_data import descendant_contexts,move_label
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
