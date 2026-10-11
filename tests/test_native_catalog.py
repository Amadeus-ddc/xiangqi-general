from collections import Counter
import json, sqlite3
import pytest
from test_paper_curriculum import source_fixture
from xqgeneral.evidence import atomic_json, manifest, digest
from xqgeneral.course_position_sampling import PositionCatalog


def fixture(tmp_path):
    roots,_,rows=source_fixture(tmp_path)
    proof=roots.parent/'recorded-manifest.json'
    flags={'complete_source_histories_and_all_native_prefix_endings_recomputed':True,
           'prior_root_future_witness_color_and_all_source_history_ownership_reserved':True,
           'prior_structured_explanation_pvs_and_branches_reserved':True,
           'prior_contexts_may_be_reused_only_in_their_existing_split':True,
           'source_history_rewritten_or_missing_past_invented':False,
           'roots_by_split':dict(Counter(r['split'] for r in rows)),'constructed_fixture':True}
    atomic_json(proof,manifest('recorded_full_native_course_root_pools',{},[],[roots],flags))
    history=roots.parent/'history-readback.json'
    atomic_json(history,manifest('recorded_full_native_course_root_readback',{},[proof,roots],[],
                {'all_source_input_and_pool_bytes_checked':True,
                 'roots_and_exclusions_regenerated':{'root':len(rows)},'constructed_fixture':True}))
    database=tmp_path/'catalog.sqlite';c=PositionCatalog(database)
    for r in rows:c.add(r)
    c.finish();c.close()
    return database,roots,proof,history,rows


def test_verified_catalog_reuses_complete_native_rows_read_only(tmp_path):
    from xqgeneral.native_catalog import verify_recorded_catalog,load_catalog
    db,roots,proof,history,rows=fixture(tmp_path);before=digest(db)
    verified=tmp_path/'verified'
    result=verify_recorded_catalog(db,[roots],[proof],[history],verified,workers=2,batch_size=2)
    assert result['roots_checked']==len(rows)
    assert result['all_native_target_masks_recomputed'] is True
    assert result['source_full_history_proof_inherited'] is True
    c=load_catalog(verified/'manifest.json',roots=[roots],source_manifests=[proof],reference_data=[])
    try:
        assert c.db.execute('SELECT count(*) FROM roots').fetchone()[0]==len(rows)
        with pytest.raises(sqlite3.OperationalError):c.db.execute('DELETE FROM roots')
    finally:c.close()
    assert digest(db)==before


@pytest.mark.parametrize('damage',['mask','value','header','extra_target'])
def test_catalog_readback_rejects_rehashed_semantic_corruption(tmp_path,damage):
    from xqgeneral.native_catalog import verify_recorded_catalog
    db,roots,proof,history,_=fixture(tmp_path)
    with sqlite3.connect(db) as c:
        if damage=='mask':c.execute('UPDATE targets SET mask=63 WHERE root=1')
        elif damage=='value':c.execute("UPDATE roots SET value='{}' WHERE id=1")
        elif damage=='header':c.execute("UPDATE roots SET split='test' WHERE id=1")
        else:c.execute('INSERT INTO targets VALUES (1,99,1,NULL,NULL)')
    with pytest.raises(ValueError,match='catalog|Catalog'):
        verify_recorded_catalog(db,[roots],[proof],[history],tmp_path/'reject',workers=1)
    assert not (tmp_path/'reject/manifest.json').exists()


def test_full_history_proof_is_required_before_trusting_stored_histories(tmp_path):
    from xqgeneral.native_catalog import verify_recorded_catalog
    db,roots,proof,history,_=fixture(tmp_path);d=json.loads(history.read_text())
    d['verification']['all_source_input_and_pool_bytes_checked']=False;atomic_json(history,d)
    with pytest.raises(ValueError,match='history|History'):
        verify_recorded_catalog(db,[roots],[proof],[history],tmp_path/'reject')
    assert not (tmp_path/'reject').exists()


def test_reuse_rejects_changed_catalog_before_rendering(tmp_path):
    from xqgeneral.native_catalog import verify_recorded_catalog,load_catalog
    db,roots,proof,history,_=fixture(tmp_path);out=tmp_path/'verified'
    verify_recorded_catalog(db,[roots],[proof],[history],out)
    with sqlite3.connect(db) as c:c.execute('UPDATE targets SET mask=63 WHERE root=1')
    with pytest.raises(ValueError,match='changed'):
        load_catalog(out/'manifest.json',roots=[roots],source_manifests=[proof],reference_data=[])


def test_complete_history_readback_must_bind_every_original_source(tmp_path):
    from xqgeneral.native_catalog import verify_recorded_catalog
    db,roots,proof,history,_=fixture(tmp_path)
    original=tmp_path/'original-game-input.json';original.write_text('{}\n')
    source=json.loads(proof.read_text())
    source['inputs'][str(original)]={'sha256':digest(original),'bytes':original.stat().st_size}
    atomic_json(proof,source)
    checked=json.loads(history.read_text())
    checked['inputs'][str(proof)]={'sha256':digest(proof),'bytes':proof.stat().st_size}
    atomic_json(history,checked)
    with pytest.raises(ValueError,match='every original input'):
        verify_recorded_catalog(db,[roots],[proof],[history],tmp_path/'reject')
    assert not (tmp_path/'reject').exists()


@pytest.mark.parametrize('workers,batch',[(0,1),(True,1),(1,0),(1,False),(1,1.5)])
def test_invalid_catalog_parallel_settings_rejected_before_output(tmp_path,workers,batch):
    from xqgeneral.native_catalog import verify_recorded_catalog
    db,roots,proof,history,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='positive workers'):
        verify_recorded_catalog(db,[roots],[proof],[history],tmp_path/'reject',workers=workers,batch_size=batch)
    assert not (tmp_path/'reject').exists()


def test_unknown_reference_isolation_cannot_be_inherited(tmp_path):
    from xqgeneral.native_catalog import verify_recorded_catalog
    db,roots,proof,history,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='reference corpus'):
        verify_recorded_catalog(db,[roots],[proof],[history],tmp_path/'reject',reference_data=[roots.parent/'reference'])
    assert not (tmp_path/'reject').exists()


def test_reuse_requires_identical_declared_source_paths(tmp_path):
    from xqgeneral.native_catalog import verify_recorded_catalog,load_catalog
    db,roots,proof,history,_=fixture(tmp_path);out=tmp_path/'verified'
    verify_recorded_catalog(db,[roots],[proof],[history],out)
    copy=tmp_path/'same-bytes.jsonl';copy.write_bytes(roots.read_bytes())
    with pytest.raises(ValueError,match='declared sources'):
        load_catalog(out/'manifest.json',roots=[copy],source_manifests=[proof],reference_data=[])


def test_native_target_only_replay_matches_already_validated_history_and_mate_witness(tmp_path):
    from xqgeneral.native_catalog import _recorded_targets
    from xqgeneral.course_position_sampling import root_targets,CLASS_BITS
    _,_,_,_,rows=fixture(tmp_path)
    for root in rows:
        if root['game_id'].endswith('mate-future-controlled-probe'):
            root=dict(root,paper_recorded_mate_witness=root['future_moves'])
        expected=[(t['horizon'],sum(CLASS_BITS[c] for c in t['classes'])) for t in root_targets(root)]
        assert _recorded_targets(root)==expected
