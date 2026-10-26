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


def controlled_prefix(tmp_path, *, early_future_selection=False, original_root=None, conditional_third=False,
                      direct_third_improvement=False):
    old = Path(original_root) if original_root is not None else completed_curriculum(tmp_path)[0]
    original_path = tmp_path / 'recipe.json'
    original = json.loads(original_path.read_text())
    original['validation_selection'] = dict(metric='current_course_raw_qa_accuracy', tie_breaker='earliest', patience=1)
    for budget in original['course_budgets']:
        budget.update(steps=8, minimum_steps=8, qa_every=8, selection_start_step=8)
    original['course_budgets'][1].update(steps=32, minimum_steps=16)
    if conditional_third:
        original['validation_selection']['patience'] = 3
        original['course_budgets'][2].update(steps=32, minimum_steps=8)
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
        actual_step = 16 if (early_future_selection or conditional_third) and index == 2 else 8
        for step in range(8, actual_step + 1, 8):
            wrong = 100000 if direct_third_improvement and index == 2 and step == 8 else 0
            raw_candidate(stage_root / 'candidates' / f'step-{step}', saved, step, actual_recipe, STAGES[:index + 1], wrong=wrong)
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


def controlled_third_boundary(tmp_path, *, original_root=None, direct=False):
    from xqgeneral.foundation_prefix import CONDITIONAL_BOUNDARY
    previous, prior_declaration, _ = controlled_prefix(tmp_path, original_root=original_root, conditional_third=True,
                                                       direct_third_improvement=direct)
    old_whole = json.loads((previous / 'manifest.json').read_text())
    old_contract = json.loads((previous / 'contract.json').read_text())
    source_recipe_path = Path(old_contract['recipe'])
    original = json.loads(source_recipe_path.read_text())
    code = old_whole['code']
    root = tmp_path / 'third-boundary-curriculum'
    recipe = dict(original, output=str(root))
    recipe_path = tmp_path / 'third-boundary-recipe.json'
    atomic_json(recipe_path, recipe)
    control_dir = tmp_path / 'third-control';control_dir.mkdir()
    entry, loop = control_dir / 'entry.py', control_dir / 'loop.py'
    entry.write_text('# controlled CPU three-course controller\n')
    loop.write_text('# controlled skip-three loop\n')
    admission_path = control_dir / 'admission.json'
    authorization_path = control_dir / 'authorization.json'
    decision_path = control_dir / 'decision.json'
    declaration_path = control_dir / 'declared-prefix.json'
    prior = json.loads(prior_declaration.read_text())
    prior_admission = json.loads(Path(prior['admission']['path']).read_text())
    artifacts = Artifacts();artifacts.json(source_recipe_path);artifacts.json(prior_declaration)
    artifacts.json(prior['admission']['path'])
    for name, binding in prior_admission['source_bound_files'].items():artifacts.add(name, binding)
    third = deepcopy(old_whole['verification']['courses'][2])
    third.pop('authorized_prefix_control')
    third.pop('imported_original_clean_first_course')
    config = artifacts.json(previous / STAGES[2] / 'training.config.json')
    history = read_history(previous / STAGES[2] / 'candidates', 16, config, code, STAGES[:3], original, artifacts)
    assert third['selection']['history'] == history and third['selection']['decision']['stop'] is False
    base, current = [row['gate']['courses'][STAGES[2]] for row in history]
    examples = base['examples'];baseline_correct = round(base['accuracy'] * examples)
    correct = round(current['accuracy'] * current['examples'])
    authorization = dict(status='explicit_latest_user_authorization', action='advance_after_no_current_course_improvement',
        stage=STAGES[2], target_step=16, baseline_selected_step=8, baseline_correct=baseline_correct,
        examples=examples, strict_improvement_required=True, no_improvement_selected_parent_step=8,
        fourth_course_recipe_preserved=True, independent_test_may_select=False)
    decision = dict(strict_improvement=False, baseline_correct=baseline_correct, baseline_step=8,
        actual_step_16_scores=dict(examples=examples, correct=correct), all_ordered_questions_identical=True,
        selected_checkpoint_if_no_improvement=third['selected_checkpoint'],
        selected_checkpoint_sha256=third['selected_checkpoint_sha256'])
    atomic_json(authorization_path, authorization);atomic_json(decision_path, decision)
    artifacts.json(authorization_path);artifacts.json(decision_path)
    third.update(completion_basis=CONDITIONAL_BOUNDARY, user_authorization=authorization,
        user_boundary_decision=decision, original_declared_minimum_stopping_policy_met=True,
        original_declared_validation_stopping_policy_met=False, original_policy_decision_preserved=True)
    if direct:
        authorization = dict(status='explicit_latest_user_authorization',
            action='select_completed_course_checkpoint_and_continue', stage=STAGES[2],
            selected_step=third['selected_step'], selected_checkpoint=third['selected_checkpoint'],
            selected_checkpoint_sha256=third['selected_checkpoint_sha256'], original_stopping_policy_met=False,
            independent_test_used=False)
        atomic_json(authorization_path, authorization)
        artifacts.expected.pop(str(authorization_path)); artifacts.json(authorization_path)
        source_log = previous / STAGES[2] / 'training/training.jsonl'
        source_log.parent.mkdir(parents=True)
        source_log.write_text(''.join(json.dumps(dict(step=step, loss=.5, gradient_norm=1., lr=.0001,
            data_batch=dict(controlled_cpu_contract=True))) + '\n' for step in range(1, 21)))
        artifacts.add(source_log, dict(sha256=digest(source_log), bytes=source_log.stat().st_size))
        stop_path = control_dir / 'discarded-source-stop.json'
        atomic_json(stop_path, dict(all_six_owned_training_processes_terminal=True,
            actual_last_completed_source_update=20, selected_step=third['selected_step'],
            selected_checkpoint_sha256=third['selected_checkpoint_sha256'],
            actual_source_training_log_sha256=digest(source_log),
            discarded_unselected_continuation_updates=4, discarded_unselected_continuation_range=[17, 20],
            controlled_cpu_contract_not_trained_model=True))
        artifacts.json(stop_path)
        third.pop('user_boundary_decision')
        third.update(completion_basis=DIRECT_BOUNDARY, user_authorization=authorization,
            retained_validated_branch_updates=16, total_source_run_completed_updates=20,
            discarded_unselected_continuation_updates=4, discarded_unselected_continuation_range=[17, 20],
            discarded_continuation_stop_verification=dict(path=str(stop_path), sha256=digest(stop_path)))
    courses = [*deepcopy(prior_admission['courses']), third]
    admission = dict(status='complete_explicit_user_boundary_three_course_prefix', original_root=str(previous),
        new_root=str(root), selected_checkpoint=third['selected_checkpoint'],
        selected_checkpoint_sha256=third['selected_checkpoint_sha256'], courses=courses,
        prior_two_course_admission=prior['admission']['path'], prior_two_course_admission_sha256=prior['admission']['sha256'],
        source_bound_files=dict(artifacts.expected), source_training_code=code,
        authorization_sha256=digest(authorization_path), entry_sha256=digest(entry), derived_loop_sha256=digest(loop),
        original_recipe_or_sources_hot_edited=False, original_patience_stopping_policy_claimed_met=False,
        independent_test_used=False)
    atomic_json(admission_path, admission)
    control = dict(admission=str(admission_path), admission_sha256=digest(admission_path), entry=str(entry),
        entry_sha256=digest(entry), derived_loop=str(loop), derived_loop_sha256=digest(loop),
        original_frozen_training_source_modified=False)
    for index, course in enumerate(courses):
        atomic_json(root / STAGES[index] / 'manifest.json', dict(status='complete',
            kind='verified_imported_foundation_course' if index == 0 else 'user_boundary_selected_foundation_course',
            verification=course, source_admission=str(admission_path), source_admission_sha256=digest(admission_path),
            original_course_directory=str(previous / STAGES[index]), training_source_code=course['training_code'],
            independent_test_used=False))
    parent = Path(third['selected_checkpoint'])
    cfg = training_config(recipe, 3, root, parent)
    fourth_root = root / STAGES[3];atomic_json(fourth_root / 'training.config.json', cfg)
    saved = torch.load(previous / STAGES[3] / 'candidates/step-8/adapter.pt', weights_only=True)
    inputs = [Path(recipe['feature_path']), Path(recipe['data_path']) / 'train.jsonl',
              Path(recipe['data_path']) / 'validation.jsonl', parent]
    saved.update(config=cfg, code=code, input_hashes={str(path): digest(path) for path in inputs})
    raw_candidate(fourth_root / 'candidates/step-8', saved, 8, recipe, STAGES)
    fourth_artifacts = Artifacts();fourth_artifacts.json(recipe_path);fourth_artifacts.json(fourth_root / 'training.config.json')
    history = read_history(fourth_root / 'candidates', 8, cfg, code, STAGES, recipe, fourth_artifacts)
    selection = selection_decision(history, recipe['validation_selection'], recipe['course_budgets'][3], stage=STAGES[3])
    assert selection['stop']
    chosen = history[0]
    fourth = dict(stage=STAGES[3], actual_steps=8, selected_step=8, global_batch_size=4, sample_presentations=32,
        gate=chosen['gate'], reference_targets_passed=chosen['gate']['passed'], selected_checkpoint=chosen['checkpoint'],
        selected_checkpoint_sha256=chosen['checkpoint_sha256'],
        selection=dict(policy=recipe['validation_selection'], budget=recipe['course_budgets'][3], decision=selection, history=history),
        imported_original_clean_first_course=False, candidate_root=str(fourth_root / 'candidates'), training_code=code,
        next_course_permitted=True, optimizer_reset_between_courses=True, independent_test_used=False, authorized_prefix_control=control)
    proof = manifest(COURSE_KIND, cfg, outputs=[chosen['checkpoint']], verification=fourth, code=code)
    proof['inputs'] = dict(fourth_artifacts.expected);atomic_json(fourth_root / 'manifest.json', proof)
    courses.append(fourth);parent = Path(chosen['checkpoint'])
    atomic_json(root / 'contract.json', dict(recipe=str(recipe_path), recipe_sha256=digest(recipe_path), code=code,
                                           authorized_prefix_control=control))
    overall = dict(courses=courses, all_courses_completed_by_declared_validation_policy=False,
        all_courses_completed_by_declared_or_user_authorized_policy=True, authorized_prefix_control=control,
        all_raw_qa_gates_passed=all(c['gate']['passed'] for c in courses), reference_targets_required_for_progression=False,
        independent_test_used=False)
    atomic_json(root / 'manifest.json', manifest(USER_CURRICULUM_KIND, recipe,
        [recipe_path, admission_path, entry, loop, *[root / s / 'manifest.json' for s in STAGES]], [parent], overall, code=code))
    atomic_json(declaration_path, dict(kind=DECLARATION_KIND, prefix_stages=list(STAGES[:3]),
        admission=dict(path=str(admission_path), sha256=digest(admission_path)),
        source_recipe=dict(path=str(source_recipe_path), sha256=digest(source_recipe_path)),
        authorization=dict(path=str(authorization_path), sha256=digest(authorization_path)),
        prior_prefix=dict(path=str(prior_declaration), sha256=digest(prior_declaration))))
    # The earlier run has only its genuine two-course prefix and raw third-course candidates.
    for path in [previous / 'manifest.json', previous / STAGES[2] / 'manifest.json', previous / STAGES[3] / 'manifest.json']:
        path.unlink()
    return root, declaration_path, parent, previous


def test_direct_improved_third_prefix_exports_selected_branch_and_accounts_for_discarded_source_work(tmp_path):
    root, declaration, selected, _ = controlled_third_boundary(tmp_path, direct=True)
    output = tmp_path / 'handoff'
    result = export_curriculum(root, output, authorized_prefix=declaration)
    third = result['courses'][2]
    assert third['selected_step'] == third['actual_steps'] == 16
    assert third['selection']['history'][0]['gate']['courses'][STAGES[2]]['accuracy'] == 0
    assert third['selection']['history'][1]['gate']['courses'][STAGES[2]]['accuracy'] == 1
    assert third['selection']['decision']['stop'] is False
    assert third['total_source_run_completed_updates'] == 20
    assert third['discarded_unselected_continuation_updates'] == 4
    assert result['courses'][1]['user_authorization']['stage'] == STAGES[1]
    assert third['user_authorization']['stage'] == STAGES[2]
    assert (output / 'adapter.pt').samefile(selected)
    reference = dict(path=str(declaration), sha256=digest(declaration))
    assert checked_parent(output / 'adapter.pt', True, authorized_prefix=reference)


@pytest.mark.parametrize('problem', ['missing_fourth', 'changed_stop', 'changed_prior_authorization'])
def test_direct_third_prefix_rejects_missing_final_course_or_changed_source_stop_provenance(tmp_path, problem):
    root, declaration, _, _ = controlled_third_boundary(tmp_path, direct=True)
    d = json.loads(declaration.read_text())
    admission = json.loads(Path(d['admission']['path']).read_text())
    if problem == 'missing_fourth':
        (root / STAGES[3] / 'manifest.json').unlink()
    elif problem == 'changed_stop':
        Path(admission['courses'][2]['discarded_continuation_stop_verification']['path']).write_text('{}')
    else:
        prior = json.loads(Path(d['prior_prefix']['path']).read_text())
        Path(prior['authorization']['path']).write_text('{}')
    output = tmp_path / 'not-created'
    with pytest.raises((ValueError, FileNotFoundError)):
        export_curriculum(root, output, authorized_prefix=declaration)
    assert not output.exists()


@pytest.mark.parametrize('problem', ['wrong_stage', 'boolean_step', 'test_selection', 'claimed_stop',
    'unaccounted_source_updates', 'wrong_discarded_range', 'boolean_source_total', 'wrong_weight', 'unmet_minimum'])
def test_direct_third_selection_rejects_false_permission_or_discarded_work_accounting(tmp_path, problem):
    from xqgeneral.foundation_prefix import validate_direct_third_selection
    root, _, _, _ = controlled_third_boundary(tmp_path, direct=True)
    third = deepcopy(json.loads((root / 'manifest.json').read_text())['verification']['courses'][2])
    authorization = deepcopy(third['user_authorization'])
    assert validate_direct_third_selection(third, authorization)['selected_step'] == 16
    if problem == 'wrong_stage': authorization['stage'] = STAGES[1]
    elif problem == 'boolean_step': authorization['selected_step'] = True
    elif problem == 'test_selection': authorization['independent_test_used'] = True
    elif problem == 'claimed_stop': third['original_declared_validation_stopping_policy_met'] = True
    elif problem == 'unaccounted_source_updates': third['discarded_unselected_continuation_updates'] = 0
    elif problem == 'wrong_discarded_range': third['discarded_unselected_continuation_range'] = [16, 20]
    elif problem == 'boolean_source_total': third['total_source_run_completed_updates'] = True
    elif problem == 'wrong_weight': authorization['selected_checkpoint_sha256'] = '0' * 64
    elif problem == 'unmet_minimum': third['original_declared_minimum_stopping_policy_met'] = False
    third['user_authorization'] = authorization
    with pytest.raises(ValueError): validate_direct_third_selection(third, authorization)


def test_direct_third_prefix_runs_sequential_sft_only_after_all_four_actual_course_proofs(tmp_path, monkeypatch):
    from test_sft_pipeline import setup_data, install_runner
    from xqgeneral.sft_pipeline import run_pipeline
    old, sft_recipe, token = setup_data(tmp_path)
    root, declaration, _, _ = controlled_third_boundary(tmp_path, original_root=old, direct=True)
    recipe = json.loads(sft_recipe.read_text())
    recipe['authorized_foundation_prefix'] = dict(path=str(declaration), sha256=digest(declaration))
    atomic_json(sft_recipe, recipe)
    runner = install_runner(monkeypatch)
    result = run_pipeline(root, 'controlled-direct-third-session', sft_recipe, token, tmp_path / 'pipeline')
    assert result['four_courses_completed_before_any_gpu_sft_work']
    assert result['initial_explanation_sft_completed'] and len(runner.jobs) == 4


def test_conditional_third_prefix_exports_all_four_native_courses_and_preserves_both_boundaries(tmp_path):
    root, declaration, selected, previous = controlled_third_boundary(tmp_path)
    output = tmp_path / 'handoff'
    result = export_curriculum(root, output, authorized_prefix=declaration)
    assert not (previous / 'manifest.json').exists()
    assert result['all_four_declared_or_authorized_selection_rules_recomputed']
    assert [c['selection']['decision']['stop'] for c in result['courses']] == [True, False, False, True]
    assert result['courses'][2]['actual_steps'] == 16 and result['courses'][2]['selected_step'] == 8
    assert result['authorized_prefix']['prefix_stages'] == list(STAGES[:3])
    assert result['authorized_prefix']['prior_prefix']['prefix_stages'] == list(STAGES[:2])
    assert (output / 'adapter.pt').samefile(selected)
    reference = dict(path=str(declaration), sha256=digest(declaration))
    assert checked_parent(output / 'adapter.pt', True, authorized_prefix=reference) == json.loads((output / 'config.json').read_text())


@pytest.mark.parametrize('problem', ['missing_fourth', 'changed_prior_authorization', 'missing_prior_prefix',
                                     'wrong_prefix_order', 'changed_authorization', 'false_stopping_claim'])
def test_conditional_third_prefix_rejects_missing_or_changed_provenance_before_export(tmp_path, problem):
    root, declaration, _, previous = controlled_third_boundary(tmp_path)
    value = json.loads(declaration.read_text())
    if problem == 'missing_fourth':
        (root / STAGES[3] / 'manifest.json').unlink()
    elif problem == 'changed_prior_authorization':
        prior = json.loads(Path(value['prior_prefix']['path']).read_text())
        Path(prior['authorization']['path']).write_text('{}')
    elif problem == 'missing_prior_prefix':
        value.pop('prior_prefix');atomic_json(declaration, value)
    elif problem == 'wrong_prefix_order':
        value['prefix_stages'].reverse();atomic_json(declaration, value)
    elif problem == 'changed_authorization':
        Path(value['authorization']['path']).write_text('{}')
    elif problem == 'false_stopping_claim':
        whole = json.loads((root / 'manifest.json').read_text())
        whole['verification']['all_courses_completed_by_declared_validation_policy'] = True
        atomic_json(root / 'manifest.json', whole)
    output = tmp_path / 'not-created'
    with pytest.raises((ValueError, FileNotFoundError)):
        export_curriculum(root, output, authorized_prefix=declaration)
    assert not output.exists()


@pytest.mark.parametrize('problem', ['wrong_metric', 'test_selection', 'boolean_target', 'changed_threshold',
                                     'wrong_checkpoint', 'claimed_original_stop', 'incomplete_history', 'wrong_stage'])
def test_conditional_boundary_recomputes_threshold_and_rejects_false_permission(tmp_path, problem):
    from xqgeneral.foundation_prefix import validate_conditional_selection
    root, _, _, _ = controlled_third_boundary(tmp_path)
    third = deepcopy(json.loads((root / 'manifest.json').read_text())['verification']['courses'][2])
    authorization = deepcopy(third['user_authorization'])
    assert validate_conditional_selection(third, authorization)['stop'] is False
    if problem == 'wrong_metric':third['selection']['policy']['metric'] = 'balanced_raw_qa_accuracy'
    elif problem == 'test_selection':authorization['independent_test_may_select'] = True
    elif problem == 'boolean_target':authorization['target_step'] = True
    elif problem == 'changed_threshold':authorization['baseline_correct'] -= 1
    elif problem == 'wrong_checkpoint':third['selected_checkpoint'] += '.different'
    elif problem == 'claimed_original_stop':third['original_declared_validation_stopping_policy_met'] = True
    elif problem == 'incomplete_history':third['selection']['history'] = third['selection']['history'][1:]
    elif problem == 'wrong_stage':third['stage'] = STAGES[1]
    third['user_authorization'] = authorization
    with pytest.raises(ValueError):validate_conditional_selection(third, authorization)


def test_conditional_third_prefix_flows_through_sequential_sft_and_locks_prior_authorization(tmp_path, monkeypatch):
    from test_sft_pipeline import setup_data, install_runner
    from xqgeneral.sft_pipeline import run_pipeline
    old, sft_recipe, token = setup_data(tmp_path)
    root, declaration, _, _ = controlled_third_boundary(tmp_path, original_root=old)
    recipe = json.loads(sft_recipe.read_text())
    recipe['authorized_foundation_prefix'] = dict(path=str(declaration), sha256=digest(declaration))
    atomic_json(sft_recipe, recipe)
    runner = install_runner(monkeypatch)
    result = run_pipeline(root, 'controlled-curriculum-session', sft_recipe, token, tmp_path / 'pipeline')
    assert result['four_courses_completed_before_any_gpu_sft_work'] and len(runner.jobs) == 4
    handoff = tmp_path / 'pipeline/handoff'
    value = json.loads(declaration.read_text())
    prior = json.loads(Path(value['prior_prefix']['path']).read_text())
    Path(prior['authorization']['path']).write_text('{}')
    with pytest.raises(ValueError):
        checked_parent(handoff / 'adapter.pt', True, authorized_prefix=recipe['authorized_foundation_prefix'])
