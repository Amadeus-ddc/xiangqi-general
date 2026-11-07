"""Native miner/collector/merge controls; certificate parsing is tested separately."""
import json
from pathlib import Path
import sys

import pytest

from test_search_collection import recorded_collection, collect_args
from test_search_pilot import setup_pilot, ControlledChain
from test_merge_search import setup_merge
from xqgeneral import collect_search, merge_search, search_distillation, search_protection
from xqgeneral.evidence import atomic_json, digest, load_jsonl, position_key
from xqgeneral.explanations import continuation_positions, parse_explanation
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.rules import legal_moves, play
from xqgeneral.symmetry import mirror_fen


def certificate(tmp_path, monkeypatch, positions, original_count):
    """Control the already-qualified certificate seam, not upstream provenance."""
    path = tmp_path / 'controlled-protection.json';atomic_json(path, dict(controlled=True))
    calls = []
    def read(path, bound, *, base_data=None, source_pool=None, **kwargs):
        assert base_data is not None and source_pool is not None
        calls.append(dict(base_data=str(base_data),source_pool=str(source_pool)))
        binding = bound.bind(path)
        note = dict(profile=search_protection.PROFILE,manifest=str(path),manifest_binding=binding,
            corpus='controlled-complete-corpus',protected_positions=len(positions),
            newly_reserved_positions=len(positions)-original_count,
            all_raw_reference_variations_and_mirrors_reconstructed=True,independent_final_test_labels_read=False)
        return set(positions), note
    monkeypatch.setattr(search_protection, 'checked_protection', read)
    return path, read, calls


def attach(miner, cert, reader, base, pool):
    bound = BoundInputs();_,note = reader(cert,bound,base_data=base,source_pool=pool)
    path = miner / 'manifest.json';proof = json.loads(path.read_text())
    proof['config']['protection_manifest'] = str(cert)
    proof['inputs'].update(bound.artifacts);proof['verification']['search_protection'] = note
    atomic_json(path, proof)
    return note


def test_native_miner_rejects_a_newly_protected_root_before_student_generation(tmp_path, monkeypatch):
    pipeline,_,config,query,_ = setup_pilot(tmp_path,monkeypatch)
    settings=json.loads(config.read_text());evaluation=json.loads(Path(settings['evaluation_config']).read_text())
    pool=Path(settings['recorded_inputs']);original=set(json.loads((pool/'heldout-positions.json').read_text()))
    key=position_key(query['record']['fen']);positions=original|{key,position_key(mirror_fen(query['record']['fen']))}
    cert,_,calls=certificate(tmp_path,monkeypatch,positions,len(original))
    native=search_distillation.SearchMiner.mine
    chain=ControlledChain(monkeypatch,query)
    monkeypatch.setattr(search_distillation.SearchMiner,'mine',native)
    output=tmp_path/'actual-miner'
    argv=['miner','--checkpoint',str(pipeline/'sft/adapter.pt'),'--data',settings['data'],
        '--features',evaluation['features'],'--recorded-inputs',str(pool),'--output',str(output),
        '--limit','1','--nodes','100000','--child-contract','move_eval','--seed',str(settings['seed']),
        '--executable',evaluation['executable'],'--weights',evaluation['weights'],
        '--protection-manifest',str(cert)]
    monkeypatch.setattr(sys,'argv',argv);search_distillation.main()
    proof=json.loads((output/'manifest.json').read_text())
    assert proof['verification']['reasons']=={'reserved_position':1}
    assert proof['verification']['accepted_for_consolidation']==0
    assert proof['inputs'][str(cert)]['sha256']==digest(cert)
    assert proof['verification']['search_protection']['newly_reserved_positions']>0
    assert len(calls)==1 and chain.teacher_loads==0


def test_collection_rechecks_new_reference_target_and_its_color_mirror(tmp_path,monkeypatch):
    args,pool,miner,query=recorded_collection(tmp_path)
    original=set(json.loads((pool/'heldout-positions.json').read_text()))
    fen=play(query['record']['fen'],query['target_fields']['branches'][1]['move'])
    extra={position_key(fen),position_key(mirror_fen(fen))}
    assert not original & extra
    cert,reader,_=certificate(tmp_path,monkeypatch,original|extra,len(original))
    attach(miner,cert,reader,args['data'],pool)
    monkeypatch.setattr(sys,'argv',collect_args(tmp_path,args,miner))
    with pytest.raises(RuntimeError,match='No real search'):collect_search.main()
    rejected=json.loads((tmp_path/'accepted/rejected.json').read_text())
    assert len(rejected)==1 and 'heldout position' in rejected[0]['reason']
    assert not (tmp_path/'accepted/manifest.json').exists()


@pytest.mark.parametrize('damage',['metadata','binding','missing_config','changed_certificate'])
def test_collection_rejects_forged_or_missing_mining_protection(tmp_path,monkeypatch,damage):
    args,pool,miner,_=recorded_collection(tmp_path)
    positions=set(json.loads((pool/'heldout-positions.json').read_text()))
    cert,reader,_=certificate(tmp_path,monkeypatch,positions,len(positions))
    attach(miner,cert,reader,args['data'],pool)
    path=miner/'manifest.json';proof=json.loads(path.read_text())
    if damage=='metadata':proof['verification']['search_protection']['protected_positions']+=1
    elif damage=='binding':proof['inputs'].pop(str(cert))
    elif damage=='missing_config':proof['config'].pop('protection_manifest')
    else:atomic_json(cert,dict(changed=True))
    atomic_json(path,proof)
    monkeypatch.setattr(sys,'argv',collect_args(tmp_path,args,miner))
    with pytest.raises(ValueError):collect_search.main()
    assert not (tmp_path/'accepted/manifest.json').exists()


def protected_merge(tmp_path,monkeypatch,*,partial=False):
    source,leaves,queries=setup_merge(tmp_path,monkeypatch)
    config=json.loads((leaves[0]/'manifest.json').read_text())['config']
    pool=Path(json.loads((leaves[0].parent/'mining/manifest.json').read_text())['config']['recorded_inputs'])
    original=set(json.loads((pool/'heldout-positions.json').read_text()));used=set(original)
    for leaf in leaves:
        for row in load_jsonl(leaf/'train.jsonl'):
            positions=continuation_positions(row,parse_explanation(row['answer']))
            used.update(positions);used.update(position_key(mirror_fen(p+' - - 0 1')) for p in positions)
    validation=load_jsonl(Path(config['validation_data'])/'validation.jsonl')[0]
    fen=validation['fen'];extra=None
    for move in sorted(legal_moves(fen)):
        child=play(fen,move);candidate={position_key(child),position_key(mirror_fen(child))}
        if not candidate & used:extra=candidate;break
    assert extra is not None
    cert,reader,_=certificate(tmp_path,monkeypatch,original|extra,len(original))
    outputs=[]
    for index,leaf in enumerate(leaves):
        if partial and index:
            outputs.append(leaf);continue
        mining=leaf.parent/'mining';attach(mining,cert,reader,config['heldout_data'],pool)
        current=json.loads((leaf/'manifest.json').read_text())['config'];output=leaf.parent/'protected-labels'
        values=dict(current,output=str(output))
        argv=['collect_search']
        for key,value in values.items():
            if type(value) is bool:
                if value:argv.append('--'+key.replace('_','-'))
            elif value is not None:
                argv.extend(['--'+key.replace('_','-'),str(value)])
        monkeypatch.setattr(sys,'argv',argv);collect_search.main();outputs.append(output)
    return source,outputs,len(original),len(extra)


def test_merge_preserves_extended_protection_and_original_authority_count(tmp_path,monkeypatch):
    source,leaves,original,extra=protected_merge(tmp_path,monkeypatch)
    output=tmp_path/'protected-merged'
    result=merge_search.merge_collections(leaves,source,output,source_profile=merge_search.SHARED_SOURCE_PROFILE)
    assert result['shared_isolation_authority']['reserved_positions']==original
    assert result['search_protection']['protected_positions']==original+extra
    proof=merge_search.collected_dataset(output,source,BoundInputs())
    assert proof['verification']==result


def test_protected_and_legacy_leaves_cannot_silently_merge(tmp_path,monkeypatch):
    source,leaves,_,_=protected_merge(tmp_path,monkeypatch,partial=True)
    output=tmp_path/'mixed-merged'
    with pytest.raises(ValueError,match='must share'):merge_search.merge_collections(leaves,source,output)
    assert not output.exists()


@pytest.mark.parametrize('empty',[False,True])
def test_extended_protection_requires_explicit_pool_and_nonempty_path_before_loading(tmp_path,monkeypatch,empty):
    argv=['miner','--checkpoint',str(tmp_path/'must-not-load.pt'),'--output',str(tmp_path/'output'),
          '--protection-manifest','' if empty else str(tmp_path/'certificate.json')]
    if empty:argv.extend(['--recorded-inputs',str(tmp_path/'pool')])
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='Extended protection'):search_distillation.main()
    assert not (tmp_path/'output').exists()
