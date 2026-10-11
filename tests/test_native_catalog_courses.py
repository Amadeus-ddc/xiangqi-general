import json,sqlite3
from pathlib import Path
import pytest
from test_native_catalog import fixture
from test_course_rendering import options
from xqgeneral.evidence import atomic_json,manifest,digest
from xqgeneral.native_catalog import verify_recorded_catalog
from xqgeneral.paper_curriculum import build,readback


def prepared(tmp_path):
    db,roots,proof,history,rows=fixture(tmp_path)
    reference=roots.parent/'reference'
    inputs=[reference/'manifest.json',*[reference/(s+'.jsonl') for s in ('train','validation','test')]]
    source=json.loads(proof.read_text())
    atomic_json(proof,manifest(source['kind'],{'reference_data':[str(reference)]},inputs,[roots],source['verification']))
    checked=json.loads(history.read_text())
    atomic_json(history,manifest(checked['kind'],{},[proof,roots,*inputs],[],checked['verification']))
    out=tmp_path/'catalog-proof'
    verify_recorded_catalog(db,[roots],[proof],[history],out,reference_data=[reference],workers=2,batch_size=2)
    return db,roots,proof,out/'manifest.json',reference


def test_verified_catalog_courses_match_fresh_selection_and_native_answers(tmp_path):
    db,roots,proof,catalog_proof,reference=prepared(tmp_path)
    old,new=tmp_path/'fresh',tmp_path/'reused'
    before=digest(db)
    serial=build([roots],[proof],old,reference_data=[reference],**options())
    reused=build([roots],[proof],new,reference_data=[reference],catalog_proof=catalog_proof,
                 question_workers=2,question_batch_size=2,**options())
    assert {k:reused[k] for k in serial}==serial
    assert reused['verified_native_source_catalog_reused'] is True
    for split in ('train','validation','test'):
        assert (old/(split+'.jsonl')).read_bytes()==(new/(split+'.jsonl')).read_bytes()
    saved=json.loads((new/'manifest.json').read_text())
    assert saved['config']['catalog_proof']==str(catalog_proof)
    assert str(db.resolve()) in saved['inputs']
    assert str(catalog_proof) in saved['inputs']
    assert readback(new,tmp_path/'readback')['questions_regenerated_and_compared']==reused['questions']
    assert digest(db)==before


def test_cached_catalog_does_not_rebuild_or_reclassify_sources(tmp_path,monkeypatch):
    import xqgeneral.paper_curriculum as module
    _,roots,proof,catalog_proof,reference=prepared(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Verified catalog must not rebuild native roots')
    monkeypatch.setattr(module,'catalog_inputs',forbidden)
    value=build([roots],[proof],tmp_path/'data',reference_data=[reference],catalog_proof=catalog_proof,**options())
    assert value['questions']>0
    assert readback(tmp_path/'data',tmp_path/'readback')['questions_regenerated_and_compared']==value['questions']


def test_reused_catalog_requires_position_profile_before_creating_corpus(tmp_path):
    _,roots,proof,catalog_proof,reference=prepared(tmp_path)
    with pytest.raises(ValueError,match='catalog'):
        build([roots],[proof],tmp_path/'reject',reference_data=[reference],catalog_proof=catalog_proof)
    assert not (tmp_path/'reject').exists()


def test_changed_catalog_rejected_before_creating_corpus(tmp_path):
    db,roots,proof,catalog_proof,reference=prepared(tmp_path)
    with sqlite3.connect(db) as connection:connection.execute('UPDATE targets SET mask=63 WHERE root=1')
    with pytest.raises(ValueError,match='changed'):
        build([roots],[proof],tmp_path/'reject',reference_data=[reference],catalog_proof=catalog_proof,**options())
    assert not (tmp_path/'reject').exists()


def test_changed_catalog_rejected_before_reading_completed_corpus(tmp_path):
    db,roots,proof,catalog_proof,reference=prepared(tmp_path)
    build([roots],[proof],tmp_path/'data',reference_data=[reference],catalog_proof=catalog_proof,**options())
    with sqlite3.connect(db) as connection:connection.execute('UPDATE targets SET mask=63 WHERE root=1')
    with pytest.raises(ValueError,match='changed'):
        readback(tmp_path/'data',tmp_path/'reject')
    assert not (tmp_path/'reject').exists()


def test_saved_catalog_contract_rejects_undeclared_profile(tmp_path):
    _,roots,proof,catalog_proof,reference=prepared(tmp_path)
    build([roots],[proof],tmp_path/'data',reference_data=[reference],catalog_proof=catalog_proof,**options())
    p=tmp_path/'data/manifest.json';saved=json.loads(p.read_text())
    saved['config'].pop('position_sampling_profile')
    for field in ('sampling_profile','task_position_budgets','task_order_profile','position_workers'):
        saved['config'].pop(field,None)
    saved['verification'].pop('task_order_profile',None)
    atomic_json(p,saved)
    with pytest.raises(ValueError,match='catalog'):
        readback(tmp_path/'data',tmp_path/'reject')
    assert not (tmp_path/'reject').exists()


@pytest.mark.parametrize('field,value',[('position_workers',True),('verified_native_source_catalog_reused',False)])
def test_saved_catalog_worker_and_reuse_claims_must_match_the_contract(tmp_path,field,value):
    _,roots,proof,catalog_proof,reference=prepared(tmp_path)
    build([roots],[proof],tmp_path/'data',reference_data=[reference],catalog_proof=catalog_proof,**options())
    path=tmp_path/'data/manifest.json';saved=json.loads(path.read_text())
    saved['config' if field=='position_workers' else 'verification'][field]=value
    atomic_json(path,saved)
    with pytest.raises(ValueError,match='workers|catalog'):
        readback(tmp_path/'data',tmp_path/'reject')
    assert not (tmp_path/'reject').exists()


def test_catalog_evidence_changed_during_readback_cannot_produce_completion(tmp_path,monkeypatch):
    import xqgeneral.paper_curriculum as module
    _,roots,proof,catalog_proof,reference=prepared(tmp_path)
    build([roots],[proof],tmp_path/'data',reference_data=[reference],catalog_proof=catalog_proof,
          question_workers=2,**options())
    original=module.render_positions
    def changed(*args,**kwargs):
        first=True
        for group in original(*args,**kwargs):
            if first:
                with catalog_proof.open('a') as handle:handle.write('\n')
                first=False
            yield group
    monkeypatch.setattr(module,'render_positions',changed)
    with pytest.raises(ValueError,match='changed'):
        readback(tmp_path/'data',tmp_path/'reject')
    assert not (tmp_path/'reject/manifest.json').exists()
