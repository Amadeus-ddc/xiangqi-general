import json
import pytest
from test_paper_curriculum import source_fixture
from xqgeneral.paper_curriculum import build, readback
from xqgeneral.course_sampling import SAMPLING_PROFILE
from xqgeneral.course_position_sampling import POSITION_PROFILE, SCARCE_FIRST_PROFILE, position_budgets
from xqgeneral.evidence import digest


def options():
    budgets=position_budgets()
    for stages in budgets.values():
        for tasks in stages.values():
            for task in tasks: tasks[task]=2
    return dict(seed=37, color_mirror=False, sampling_profile=SAMPLING_PROFILE,
                position_sampling_profile=POSITION_PROFILE, task_position_budgets=budgets,
                task_order_profile=SCARCE_FIRST_PROFILE)


def test_parallel_native_render_keeps_adaptive_queries_bytes_order_and_readback(tmp_path):
    roots,proof,_=source_fixture(tmp_path)
    old,new=tmp_path/'serial',tmp_path/'parallel'
    before=digest(roots),digest(proof)
    reference=[roots.parent/'reference']
    serial=build([roots],[proof],old,reference_data=reference,**options())
    parallel=build([roots],[proof],new,reference_data=reference,question_workers=2,
                   question_batch_size=1,**options())
    assert serial==parallel
    for split in ('train','validation','test'):
        assert (old/f'{split}.jsonl').read_bytes()==(new/f'{split}.jsonl').read_bytes()
    assert before==(digest(roots),digest(proof))
    assert readback(new,tmp_path/'parallel-readback')['questions_regenerated_and_compared']==parallel['questions']
    saved=json.loads((new/'manifest.json').read_text())
    assert saved['config']['question_workers']==2
    assert saved['config']['question_batch_size']==1


@pytest.mark.parametrize('field,value',[('question_workers',True),('question_workers',0),
    ('question_workers',2.5),('question_batch_size',False),('question_batch_size',0)])
def test_invalid_renderer_settings_rejected_before_creating_corpus(tmp_path,field,value):
    roots,proof,_=source_fixture(tmp_path)
    target=tmp_path/'reject'
    with pytest.raises(ValueError,match='question'):
        build([roots],[proof],target,**{field:value},**options())
    assert not target.exists()


def test_parallel_renderer_requires_declared_position_sampling_without_color_mirroring(tmp_path):
    roots,proof,_=source_fixture(tmp_path)
    for change in ({'color_mirror':True},{'position_sampling_profile':None,'task_position_budgets':None,
                                          'task_order_profile':None}):
        args=options();args.update(change)
        with pytest.raises(ValueError,match='question'):
            build([roots],[proof],tmp_path/'reject',question_workers=2,**args)
        assert not (tmp_path/'reject').exists()


def test_legacy_serial_manifest_omits_new_worker_settings(tmp_path):
    roots,proof,_=source_fixture(tmp_path)
    data=tmp_path/'legacy'
    build([roots],[proof],data,**options())
    config=json.loads((data/'manifest.json').read_text())['config']
    assert 'question_workers' not in config and 'question_batch_size' not in config


def test_parallel_worker_rejects_forged_native_history_instead_of_emitting_drafts(tmp_path):
    from xqgeneral.course_rendering import render_positions
    from xqgeneral.course_sampling import make_sampler
    roots, _, originals = source_fixture(tmp_path)
    root = dict(originals[0], feature_key='forged-unvalidated-history')
    with pytest.raises(ValueError, match='native full-history'):
        list(render_positions([(root, 'static_current', 'piece')], 37,
                              make_sampler(SAMPLING_PROFILE), workers=2))


@pytest.mark.parametrize('field,value',[('question_workers',True),('question_batch_size',0)])
def test_readback_rejects_invalid_saved_renderer_contract_before_output(tmp_path,field,value):
    roots,proof,_=source_fixture(tmp_path)
    data=tmp_path/'corpus'
    build([roots],[proof],data,**options())
    p=data/'manifest.json'; proof=json.loads(p.read_text()); proof['config'][field]=value
    p.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='question'):
        readback(data,tmp_path/'rejected-readback')
    assert not (tmp_path/'rejected-readback').exists()
