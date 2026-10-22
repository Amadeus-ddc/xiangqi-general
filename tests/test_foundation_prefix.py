"""Real native-QA/CPU tensor integration; the controlled weights were never trained."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_curriculum_selection import raw_candidate
from test_foundation_handoff import completed_curriculum
from xqgeneral.curriculum_data import STAGES
from xqgeneral.curriculum_selection import COURSE_KIND, USER_CURRICULUM_KIND, USER_HANDOFF_KIND, read_history, selection_decision
from xqgeneral.evidence import atomic_json, code_identity, digest, manifest
from xqgeneral.foundation_handoff import Artifacts, export_curriculum
from xqgeneral.foundation_prefix import DECLARATION_KIND, DIRECT_BOUNDARY, declared_prefix_reference, validate_direct_selection
from xqgeneral.gated_curriculum import training_config
from xqgeneral.sft import checked_parent


def controlled_prefix(tmp_path, *, early_future_selection=False, original_root=None):
    old = Path(original_root) if original_root is not None else completed_curriculum(tmp_path)[0]
    original_path = tmp_path / 'recipe.json'
    original = json.loads(original_path.read_text())
    original['validation_selection'] = dict(metric='current_course_raw_qa_accuracy', tie_breaker='earliest', patience=1)
    for budget in original['course_budgets']:
        budget.update(steps=8, minimum_steps=8, qa_every=8, selection_start_step=8)
    original['course_budgets'][1].update(steps=32, minimum_steps=16)
    if early_future_selection:
        original['course_budgets'][2].update(steps=16, minimum_steps=16)
    atomic_json(original_path, original)
    root = tmp_path / 'user-curriculum'
    recipe = dict(original, output=str(root))
    recipe_path = tmp_path / 'user-recipe.json'
    atomic_json(recipe_path, recipe)
    control_dir = tmp_path / 'control';control_dir.mkdir()
    entry, loop = control_dir / 'entry.py', control_dir / 'loop.py'
    entry.write_text('# controlled CPU prefix controller\n');loop.write_text('# controlled skip-two loop\n')
    admission_path = control_dir / 'admission.json'
    authorization_path = control_dir / 'authorization.json'
    declaration_path = control_dir / 'declared-prefix.json'
    code = code_identity()
    artifacts = Artifacts();artifacts.json(original_path)
    results = [];parent = None;authorization = None
    for index, stage in enumerate(STAGES):
        producer = old if index < 2 else root
        actual_recipe = original if index < 2 else recipe
        config = training_config(actual_recipe, index, producer, parent)
        stage_root = producer / stage
        atomic_json(stage_root / 'training.config.json', config)
        original_saved = torch.load(old / stage / 'candidates/step-8/adapter.pt', weights_only=True)
        inputs = [Path(recipe['feature_path']), Path(recipe['data_path']) / 'train.jsonl', Path(recipe['data_path']) / 'validation.jsonl']
        if parent:inputs.append(parent)
        saved = dict(original_saved, config=config, code=code, input_hashes={str(p): digest(p) for p in inputs})
        actual_step = 16 if early_future_selection and index == 2 else 8
        for step in range(8, actual_step + 1, 8):
            raw_candidate(stage_root / 'candidates' / f'step-{step}', saved, step, actual_recipe, STAGES[:index + 1])
        local = artifacts if index < 2 else Artifacts()
        local.json(original_path if index < 2 else recipe_path)
        local.json(stage_root / 'training.config.json')
        history = read_history(stage_root / 'candidates', actual_step, config, code, STAGES[:index + 1], actual_recipe, local)
        decision = selection_decision(history, recipe['validation_selection'], recipe['course_budgets'][index], stage=stage)
        chosen = next(row for row in history if row['step'] == decision['selected_step'])
        result = dict(stage=stage, actual_steps=actual_step, selected_step=chosen['step'], global_batch_size=4,
            sample_presentations=actual_step * 4, gate=chosen['gate'], reference_targets_passed=chosen['gate']['passed'],
            selected_checkpoint=chosen['checkpoint'], selected_checkpoint_sha256=chosen['checkpoint_sha256'],
            selection=dict(policy=recipe['validation_selection'], budget=recipe['course_budgets'][index], decision=decision, history=history),
            imported_original_clean_first_course=False, candidate_root=str(stage_root / 'candidates'), training_code=code,
            next_course_permitted=True, optimizer_reset_between_courses=True, independent_test_used=False)
        if index == 1:
            assert not decision['stop']
            authorization = dict(status='explicit_latest_user_authorization', action='select_completed_course_checkpoint_and_continue',
                stage=stage, selected_step=chosen['step'], selected_checkpoint=chosen['checkpoint'],
                selected_checkpoint_sha256=chosen['checkpoint_sha256'], original_stopping_policy_met=False, independent_test_used=False)
            atomic_json(authorization_path, authorization)
            result.update(completion_basis=DIRECT_BOUNDARY, user_authorization=authorization,
                          original_declared_minimum_stopping_policy_met=False, original_policy_decision_preserved=True)
        elif index >= 2:
            result['authorized_prefix_control'] = control
        results.append(result)
        if index == 0:
            proof = manifest(COURSE_KIND, config, outputs=[chosen['checkpoint']], verification=result, code=code)
            proof['inputs'] = dict(local.expected)
            atomic_json(stage_root / 'manifest.json', proof)
            artifacts.proof(stage_root / 'manifest.json', COURSE_KIND)
        elif index == 1:
            artifacts.json(authorization_path)
            admission = dict(status='complete_explicit_user_boundary_two_course_prefix', original_root=str(old),new_root=str(root),
                selected_checkpoint=chosen['checkpoint'],selected_checkpoint_sha256=chosen['checkpoint_sha256'],courses=results[:],
                original_course1_manifest_sha256=digest(old / STAGES[0] / 'manifest.json'), source_bound_files=dict(artifacts.expected),
                source_training_code=code, authorization_sha256=digest(authorization_path), entry_sha256=digest(entry), derived_loop_sha256=digest(loop),
                original_recipe_or_sources_hot_edited=False,original_stopping_policy_claimed_met=False,independent_test_used=False)
            atomic_json(admission_path, admission)
            control = dict(admission=str(admission_path),admission_sha256=digest(admission_path),entry=str(entry),entry_sha256=digest(entry),
                derived_loop=str(loop),derived_loop_sha256=digest(loop),original_frozen_training_source_modified=False)
            for j in range(2):
                atomic_json(root / STAGES[j] / 'manifest.json', dict(status='complete',
                    kind='verified_imported_foundation_course' if j == 0 else 'user_boundary_selected_foundation_course',
                    verification=results[j],source_admission=str(admission_path),source_admission_sha256=digest(admission_path),
                    original_course_directory=str(old / STAGES[j]),training_source_code=code,independent_test_used=False))
        else:
            proof = manifest(COURSE_KIND, config, outputs=[chosen['checkpoint']],verification=result,code=code)
            proof['inputs'] = dict(local.expected)
            atomic_json(stage_root / 'manifest.json', proof)
        parent = Path(chosen['checkpoint'])
    atomic_json(root / 'contract.json', dict(recipe=str(recipe_path),recipe_sha256=digest(recipe_path),code=code,authorized_prefix_control=control))
    overall = dict(courses=results,all_courses_completed_by_declared_validation_policy=False,
        all_courses_completed_by_declared_or_user_authorized_policy=True,authorized_prefix_control=control,
        all_raw_qa_gates_passed=all(r['gate']['passed'] for r in results),reference_targets_required_for_progression=False,independent_test_used=False)
    atomic_json(root / 'manifest.json',manifest(USER_CURRICULUM_KIND,recipe,
        [recipe_path,admission_path,entry,loop,*[root / s / 'manifest.json' for s in STAGES]],[parent],overall,code=code))
    atomic_json(declaration_path,dict(kind=DECLARATION_KIND,prefix_stages=list(STAGES[:2]),
        admission=dict(path=str(admission_path),sha256=digest(admission_path)),
        source_recipe=dict(path=str(original_path),sha256=digest(original_path)),
        authorization=dict(path=str(authorization_path),sha256=digest(authorization_path))))
    return root,declaration_path,parent


def test_authorized_prefix_exports_real_native_qa_and_exact_weights_then_requires_consumer_opt_in(tmp_path):
    root,declaration,selected=controlled_prefix(tmp_path)
    output=tmp_path/'handoff'
    result=export_curriculum(root,output,authorized_prefix=declaration)
    assert result['all_four_declared_or_authorized_selection_rules_recomputed']
    assert result['all_four_validation_selection_rules_recomputed'] is False
    assert result['courses'][1]['selection']['decision']['stop'] is False
    assert (output/'adapter.pt').samefile(selected)
    assert json.loads((output/'manifest.json').read_text())['kind']==USER_HANDOFF_KIND
    with pytest.raises(ValueError,match='explicitly opt in'):checked_parent(output/'adapter.pt',True)
    reference=dict(path=str(declaration),sha256=digest(declaration))
    assert checked_parent(output/'adapter.pt',True,authorized_prefix=reference)==json.loads((output/'config.json').read_text())


def test_normal_current_course_selector_can_export_a_future_checkpoint_before_its_stopping_minimum(tmp_path):
    root,declaration,_=controlled_prefix(tmp_path,early_future_selection=True)
    result=export_curriculum(root,tmp_path/'handoff',authorized_prefix=declaration)
    assert result['courses'][2]['actual_steps']==16 and result['courses'][2]['selected_step']==8


@pytest.mark.parametrize('problem',['no_opt_in','incomplete','missing_fourth','changed_authorization','changed_weight','claimed_original_stop','wrong_prefix_order'])
def test_incomplete_or_changed_authorized_prefix_never_creates_an_export(tmp_path,problem):
    root,declaration,selected=controlled_prefix(tmp_path)
    if problem=='incomplete':
        p=root/'manifest.json';v=json.loads(p.read_text());v['verification']['courses'].pop();atomic_json(p,v)
    elif problem=='missing_fourth':(root/STAGES[3]/'manifest.json').unlink()
    elif problem=='changed_authorization':
        d=json.loads(declaration.read_text());Path(d['authorization']['path']).write_text('{}')
    elif problem=='changed_weight':
        with selected.open('ab')as f:f.write(b'changed')
    elif problem=='claimed_original_stop':
        p=root/'manifest.json';v=json.loads(p.read_text());v['verification']['all_courses_completed_by_declared_validation_policy']=True;atomic_json(p,v)
    elif problem=='wrong_prefix_order':
        v=json.loads(declaration.read_text());v['prefix_stages'].reverse();atomic_json(declaration,v)
    output=tmp_path/'not-created'
    with pytest.raises((ValueError,FileNotFoundError)):
        export_curriculum(root,output,**({}if problem=='no_opt_in'else{'authorized_prefix':declaration}))
    assert not output.exists()


def test_direct_boundary_does_not_accept_a_nonbest_choice_or_a_false_authorization(tmp_path):
    root,declaration,_=controlled_prefix(tmp_path)
    second=json.loads((root/'manifest.json').read_text())['verification']['courses'][1]
    authorization=second['user_authorization']
    assert validate_direct_selection(second,authorization)['stop'] is False
    for changed in ['selected_step','independent_test_used','original_stopping_policy_met','action']:
        bad=deepcopy(authorization);bad[changed]=True
        with pytest.raises(ValueError):validate_direct_selection(second,bad)


def test_locked_prefix_reference_rejects_changes_before_pipeline_or_gpu_work(tmp_path):
    root,declaration,_=controlled_prefix(tmp_path)
    reference=dict(path=str(declaration),sha256=digest(declaration))
    assert declared_prefix_reference({'authorized_foundation_prefix':reference})==reference
    declaration.write_text('{}')
    with pytest.raises(ValueError):declared_prefix_reference({'authorized_foundation_prefix':reference})


def test_authorized_prefix_flows_through_the_actual_sequential_sft_contract(tmp_path,monkeypatch):
    from test_sft_pipeline import setup_data,install_runner
    from xqgeneral.sft_pipeline import run_pipeline
    old,sft_recipe,token=setup_data(tmp_path)
    root,declaration,_=controlled_prefix(tmp_path,original_root=old)
    recipe=json.loads(sft_recipe.read_text())
    recipe['authorized_foundation_prefix']=dict(path=str(declaration),sha256=digest(declaration))
    atomic_json(sft_recipe,recipe)
    runner=install_runner(monkeypatch)
    result=run_pipeline(root,'controlled-curriculum-session',sft_recipe,token,tmp_path/'pipeline')
    assert result['four_courses_completed_before_any_gpu_sft_work']
    assert result['initial_explanation_sft_completed'] and len(runner.jobs)==4
    handoff=json.loads((tmp_path/'pipeline/handoff/manifest.json').read_text())
    assert handoff['kind']==USER_HANDOFF_KIND
    assert handoff['verification']['courses'][1]['selection']['decision']['stop'] is False


def test_pipeline_without_explicit_prefix_permission_never_starts_a_model(tmp_path,monkeypatch):
    from test_sft_pipeline import setup_data,install_runner
    from xqgeneral.sft_pipeline import run_pipeline
    old,sft_recipe,token=setup_data(tmp_path)
    root,_,_=controlled_prefix(tmp_path,original_root=old)
    runner=install_runner(monkeypatch)
    with pytest.raises(ValueError,match='completed four-course'):
        run_pipeline(root,'controlled-curriculum-session',sft_recipe,token,tmp_path/'pipeline')
    assert runner.jobs==[]


@pytest.mark.parametrize('problem',['wrong_metric','boolean_step'])
def test_direct_boundary_rejects_an_ineligible_metric_or_boolean_authorization_step(tmp_path,problem):
    root,_,_=controlled_prefix(tmp_path)
    second=json.loads((root/'manifest.json').read_text())['verification']['courses'][1]
    authorization=deepcopy(second['user_authorization'])
    if problem=='wrong_metric':second['selection']['policy']['metric']='balanced_raw_qa_accuracy'
    else:authorization['selected_step']=True
    with pytest.raises(ValueError):validate_direct_selection(second,authorization)
