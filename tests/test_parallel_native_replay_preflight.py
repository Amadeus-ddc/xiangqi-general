"""Replay CPU work must honor the caller's workers without duplicate grading."""
from collections import Counter
import json
from pathlib import Path

import pytest

import test_auxiliary_search_replay as replay_control
import test_distillation_joint_replay as joint_control
from xqgeneral import auxiliary_search_replay as replay, distillation_preflight
from xqgeneral.evidence import atomic_json, digest, load_jsonl, write_jsonl
from xqgeneral.feature_store import FeatureStoreWriter
from xqgeneral.recorded_search_inputs import BoundInputs
from test_feature_store import block


def native_gate(tmp_path, monkeypatch):
    args, _, _, _, _ = joint_control.native_replay(tmp_path, monkeypatch)
    data = replay_control.dataset(tmp_path/'auxiliary-source')
    subset = tmp_path/'six-task-replay'; replay.prepare(data, subset, train_per_stage=2)
    rows = [*load_jsonl(subset/'train.jsonl'),*load_jsonl(subset/'validation.jsonl')]
    old_cache = Path(args[3]);base = Path(json.loads(old_cache.read_text())['config']['base_cache'])
    writer=FeatureStoreWriter(tmp_path/'six-task-features',[0,3],'a'*64,
        config=dict(base_cache=str(base),base_proof=str(base)))
    writer.add_existing(base,base)
    def keys(path):
        return set(json.loads(Path(json.loads(path.read_text())['catalog']['keys']).read_text()))
    writer.append(block(sorted(({row['feature_key'] for row in rows}|keys(old_cache))-keys(base)),width=512))
    writer.finish()
    return [*args[:3],writer.output/'manifest.json',args[4]],subset,rows


def test_requested_native_preflight_workers_reach_auxiliary_replay_validation(tmp_path,monkeypatch):
    args,subset,_=native_gate(tmp_path,monkeypatch)
    class ObservedWorkers(Exception):
        def __init__(self,workers):self.workers=workers
    def observe(directory,bound,*,workers=1,profile=replay.PROFILE):
        assert profile == replay.PROFILE
        raise ObservedWorkers(workers)
    monkeypatch.setattr(replay,'checked_subset',observe)
    with pytest.raises(ObservedWorkers) as seen:
        distillation_preflight.preflight(*args,tmp_path/'gate',workers=4,
            replay_data_paths=[subset],replay_profile=replay.PROFILE)
    assert seen.value.workers==4,'Replay grading silently ignored requested preflight parallelism'


def test_native_preflight_regrades_every_auxiliary_row_once_with_complete_original_coverage(tmp_path,monkeypatch):
    args,subset,rows=native_gate(tmp_path,monkeypatch)
    original=replay.native_label;seen=[]
    def observe(row):
        seen.append(row['id']);return original(row)
    monkeypatch.setattr(replay,'native_label',observe)
    result=distillation_preflight.preflight(*args,tmp_path/'gate',workers=1,
        replay_data_paths=[subset],replay_profile=replay.PROFILE)
    assert Counter(seen)==Counter({row['id']:1 for row in rows}), 'Native replay answers were regraded twice'
    assert result['course_replay']['train_validation_rows_tokenized']==len(rows)
    assert result['course_replay']['all_raw_train_validation_answers_natively_regraded']
    assert result['protected_validation_and_test_bytes_unchanged']
    assert result['independent_test_rows_tokenized']==0


def test_parallel_native_selection_and_legacy_subset_reopening_preserve_every_raw_byte(tmp_path):
    data=replay_control.dataset(tmp_path/'source')
    serial,serial_proof=replay.selection(data,2,71,BoundInputs())
    parallel,parallel_proof=replay.selection(data,2,71,BoundInputs(),workers=2)
    assert serial==parallel and serial_proof==parallel_proof
    subset=tmp_path/'subset';replay.prepare(data,subset,train_per_stage=2,seed=71)
    files=[subset/name for name in ['manifest.json','train.jsonl','validation.jsonl','test.jsonl']]
    before={p:(digest(p),p.stat().st_mtime_ns) for p in files}
    rows,proof=replay.checked_subset(subset,BoundInputs(),workers=2)
    assert rows==[row for values in serial.values() for _,row in values]
    assert proof==serial_proof
    assert all((digest(p),p.stat().st_mtime_ns)==value for p,value in before.items())


@pytest.mark.parametrize('workers',[1,2])
def test_rehashed_source_and_identical_subset_cannot_bypass_actual_native_answer_grading(tmp_path,workers):
    data=replay_control.dataset(tmp_path/'source');subset=tmp_path/'subset'
    replay.prepare(data,subset,train_per_stage=2)
    selected=load_jsonl(subset/'train.jsonl')
    target=next(row for row in selected if row['stage'] in replay.COURSES)
    target_id=target['id']
    original=load_jsonl(data/'train.jsonl')
    for rows in [original,selected]:
        next(row for row in rows if row['id']==target_id)['answer']='controlled incorrect native answer'
    write_jsonl(data/'train.jsonl',original);write_jsonl(subset/'train.jsonl',selected)
    def binding(p):return dict(sha256=digest(p),bytes=p.stat().st_size)
    p=data/'manifest.json';proof=json.loads(p.read_text())
    proof['outputs'][str(data/'train.jsonl')]=binding(data/'train.jsonl');atomic_json(p,proof)
    p=subset/'manifest.json';proof=json.loads(p.read_text())
    for source in [data/'manifest.json',data/'train.jsonl']:
        assert str(source) in proof['inputs'];proof['inputs'][str(source)]=binding(source)
    proof['outputs'][str(subset/'train.jsonl')]=binding(subset/'train.jsonl');atomic_json(p,proof)
    with pytest.raises(ValueError,match='Replay answer differs from its native course rules'):
        replay.checked_subset(subset,BoundInputs(),workers=workers)


@pytest.mark.parametrize('workers',[0,-1,True,1.5])
@pytest.mark.parametrize('method',['selection','checked_subset'])
def test_invalid_replay_worker_budgets_fail_before_reading_sources(tmp_path,workers,method):
    with pytest.raises(ValueError,match='workers must be a positive integer'):
        if method=='selection':replay.selection(tmp_path/'absent',2,71,BoundInputs(),workers=workers)
        else:replay.checked_subset(tmp_path/'absent',BoundInputs(),workers=workers)
