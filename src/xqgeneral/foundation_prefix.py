"""Verify explicit ordered course boundaries without changing their stopping history."""
from pathlib import Path
import json
import math

from .curriculum_data import STAGES
from .curriculum_selection import COURSE_KIND, CURRENT_COURSE_METRIC, course_contract, selection_decision
from .evidence import digest
from .foundation_preflight import clean_recipe


DECLARATION_KIND = 'explicit_user_authorized_course_prefix'
DIRECT_BOUNDARY = 'explicit_user_authorized_direct_course_boundary'
CONDITIONAL_BOUNDARY = 'explicit_user_authorized_conditional_boundary'


def declared_prefix_reference(recipe):
    """Require the caller's locked declaration, rather than infer permission from model artifacts."""
    value = recipe.get('authorized_foundation_prefix')
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'} or
            not isinstance(value['path'], str) or not isinstance(value['sha256'], str) or
            digest(value['path']) != value['sha256']):
        raise ValueError('Bind the explicit foundation-prefix declaration by path and exact SHA256')
    return value


def pinned_json(artifacts, binding):
    if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'} or
            not isinstance(binding['path'], str) or not isinstance(binding['sha256'], str)):
        raise ValueError('Declare a prefix artifact path and its exact SHA256')
    path = Path(binding['path'])
    artifacts.add(path, {'sha256': binding['sha256'], 'bytes': path.stat().st_size})
    return artifacts.json(path)


def validate_direct_selection(result, authorization):
    """Recompute the best checkpoint and retain the original unmet stopping decision."""
    selection = result['selection']
    if result['stage'] != STAGES[1] or selection['policy']['metric'] != CURRENT_COURSE_METRIC:
        raise ValueError('The early second-course boundary requires its original current-course raw validation metric')
    actual = selection_decision(selection['history'], selection['policy'], selection['budget'], stage=result['stage'])
    if actual['selected_step'] is None:
        raise ValueError('The user boundary must select an actual eligible raw-validation checkpoint')
    chosen = next(row for row in selection['history'] if row['step'] == actual['selected_step'])
    action = authorization.get('action')
    explicit_action = (action == 'select_completed_course_checkpoint_and_continue' and
                       authorization.get('stage') == STAGES[1])
    legacy_action = action == f"select_completed_second_course_{result['selected_step']}_and_start_course3"
    minimum_marker = f"original_second_course_minimum_{selection['budget']['minimum_steps']}_not_claimed_met"
    unmet_rule_preserved = (authorization.get('original_stopping_policy_met') is False or
                           authorization.get(minimum_marker) is True)
    if (result['stage'] != STAGES[1] or result.get('completion_basis') != DIRECT_BOUNDARY or
            result.get('original_declared_minimum_stopping_policy_met') is not False or
            result.get('original_policy_decision_preserved') is not True or actual['stop'] or
            selection['decision'] != actual or result['actual_steps'] != actual['steps_completed'] or
            result['actual_steps'] >= selection['budget']['minimum_steps'] or
            result['selected_step'] != chosen['step'] or result['selected_checkpoint'] != chosen['checkpoint'] or
            result['selected_checkpoint_sha256'] != chosen['checkpoint_sha256'] or result['gate'] != chosen['gate'] or
            result.get('reference_targets_passed') != chosen['gate']['passed'] or
            result.get('user_authorization') != authorization or
            authorization.get('status') != 'explicit_latest_user_authorization' or
            not (explicit_action or legacy_action) or
            authorization.get('selected_step') != result['selected_step'] or
            type(authorization.get('selected_step')) is not int or
            authorization.get('selected_checkpoint') != result['selected_checkpoint'] or
            authorization.get('selected_checkpoint_sha256') != result['selected_checkpoint_sha256'] or
            not unmet_rule_preserved or
            authorization.get('independent_test_used') is not False):
        raise ValueError('The explicit user boundary must preserve the actual best validation choice and unmet stopping rule')
    return actual


def validate_conditional_selection(result, authorization):
    """Allow the third-course boundary only after an actual failure to improve."""
    selection = result['selection']
    stage = STAGES[2]
    if result['stage'] != stage or selection['policy']['metric'] != CURRENT_COURSE_METRIC:
        raise ValueError('A conditional third-course boundary requires current-course raw validation')
    history = selection['history']
    actual = selection_decision(history, selection['policy'], selection['budget'], stage=stage)
    if len(history) < 2 or actual['selected_step'] is None:
        raise ValueError('A conditional boundary requires a prior eligible checkpoint and a new validation')
    before = selection_decision(history[:-1], selection['policy'], selection['budget'], stage=stage)
    if before['selected_step'] is None:
        raise ValueError('The conditional boundary has no eligible prior best checkpoint')
    best = next(row for row in history[:-1] if row['step'] == before['selected_step'])
    chosen = next(row for row in history if row['step'] == actual['selected_step'])
    baseline, current = best['gate']['courses'][stage], history[-1]['gate']['courses'][stage]
    examples = baseline['examples']
    correct_before = round(baseline['accuracy'] * examples)
    correct_now = round(current['accuracy'] * current['examples'])
    target = actual['steps_completed']
    decision = result.get('user_boundary_decision', {})
    counts = decision.get(f'actual_step_{target}_scores', {})
    explicit = (authorization.get('status') == 'explicit_latest_user_authorization' and
                authorization.get('action') == 'advance_after_no_current_course_improvement' and
                authorization.get('stage') == stage)
    legacy = (authorization.get('action') ==
              'conditional_course3_stop_at26000_without_strict_current_course_improvement_and_start_course4' and
              target == 26000 and before['selected_step'] == 24000 and examples == 700 and correct_before == 690 and
              authorization.get('user_instruction') == '2.6w步如果没有提升直接进入第四课')
    if (not (explicit or legacy) or actual['stop'] or selection['decision'] != actual or
            result.get('completion_basis') != CONDITIONAL_BOUNDARY or
            result.get('original_declared_minimum_stopping_policy_met') is not True or
            result.get('original_declared_validation_stopping_policy_met') is not False or
            result.get('original_policy_decision_preserved') is not True or
            result['actual_steps'] != target or target < selection['budget']['minimum_steps'] or
            result['selected_step'] != before['selected_step'] or result['selected_step'] != chosen['step'] or
            result['selected_checkpoint'] != chosen['checkpoint'] or
            result['selected_checkpoint_sha256'] != chosen['checkpoint_sha256'] or
            result['gate'] != chosen['gate'] or result.get('reference_targets_passed') != chosen['gate']['passed'] or
            result.get('user_authorization') != authorization or
            any(type(authorization.get(key)) is not int for key in
                ['target_step', 'baseline_selected_step', 'baseline_correct', 'examples', 'no_improvement_selected_parent_step']) or
            authorization['target_step'] != target or authorization['baseline_selected_step'] != before['selected_step'] or
            authorization['no_improvement_selected_parent_step'] != before['selected_step'] or
            authorization['baseline_correct'] != correct_before or authorization['examples'] != examples or
            authorization.get('strict_improvement_required') is not True or
            authorization.get('fourth_course_recipe_preserved') is not True or
            authorization.get('independent_test_may_select') is not False or
            current['examples'] != examples or correct_now > correct_before or
            not math.isclose(baseline['accuracy'] * examples, correct_before, rel_tol=0., abs_tol=1e-9) or
            not math.isclose(current['accuracy'] * examples, correct_now, rel_tol=0., abs_tol=1e-9) or
            decision.get('strict_improvement') is not False or decision.get('baseline_correct') != correct_before or
            decision.get('baseline_step') != before['selected_step'] or
            counts.get('examples') != examples or counts.get('correct') != correct_now or
            not (decision.get('all_ordered_questions_identical') is True or
                 decision.get('all2000_ordered_questions_identical') is True) or
            decision.get('selected_checkpoint_if_no_improvement') != chosen['checkpoint'] or
            decision.get('selected_checkpoint_sha256') != chosen['checkpoint_sha256'] or
            result.get('independent_test_used') is not False):
        raise ValueError('The conditional boundary must retain the actual prior best after no strict improvement')
    return actual


class AuthorizedPrefix:
    """Resolve original prefix artifacts; later courses still use their complete ordinary proofs."""
    def __init__(self, artifacts, declaration_path, root, recipe, whole):
        self.artifacts = artifacts
        self.path = Path(declaration_path)
        self.declaration = declaration = artifacts.json(self.path)
        stages = declaration.get('prefix_stages')
        self.count = len(stages) if isinstance(stages, list) else 0
        fields = {'kind', 'prefix_stages', 'admission', 'source_recipe', 'authorization'}
        if self.count == 3:
            fields.add('prior_prefix')
        if (self.count not in (2, 3) or set(declaration) != fields or
                declaration['kind'] != DECLARATION_KIND or stages != list(STAGES[:self.count])):
            raise ValueError('Opt in to an exact ordered two- or three-course prefix declaration')
        self.admission = admission = pinned_json(artifacts, declaration['admission'])
        self.recipe = original = pinned_json(artifacts, declaration['source_recipe'])
        self.authorization = pinned_json(artifacts, declaration['authorization'])
        self.original_root = Path(admission['original_root'])
        self.root = Path(root)
        self.code = whole['code']
        clean_recipe(original)
        old_minimum = original['course_budgets'][1]['minimum_steps']
        stopping_claim = (admission.get('original_stopping_policy_claimed_met',
                                       admission.get(f'old_{old_minimum}_stopping_requirement_claimed_met'))
                          if self.count == 2 else admission.get('original_patience_stopping_policy_claimed_met'))
        expected_status = ('complete_explicit_user_boundary_two_course_prefix' if self.count == 2 else
                           'complete_explicit_user_boundary_three_course_prefix')
        if (admission.get('status') != expected_status or
                Path(admission['new_root']).resolve() != self.root.resolve() or
                self.original_root.resolve() == self.root.resolve() or
                Path(original['output']).resolve() != self.original_root.resolve() or
                dict(original, output=recipe['output']) != recipe or len(admission.get('courses', [])) != self.count or
                admission.get('source_training_code') != self.code or
                admission.get('original_recipe_or_sources_hot_edited') is not False or
                stopping_claim is not False or
                admission.get('independent_test_used') is not False or
                admission.get('authorization_sha256') != declaration['authorization']['sha256']):
            raise ValueError('Authorized prefix must bind its preserved recipe, source, authorization and actual courses')
        for name, binding in admission['source_bound_files'].items():
            artifacts.add(name, binding)
        control = whole['verification'].get('authorized_prefix_control', {})
        expected_control = {'admission', 'admission_sha256', 'entry', 'entry_sha256', 'derived_loop',
                            'derived_loop_sha256', 'original_frozen_training_source_modified'}
        if (set(control) != expected_control or
                control['admission'] != declaration['admission']['path'] or
                control['admission_sha256'] != declaration['admission']['sha256'] or
                control['entry_sha256'] != admission['entry_sha256'] or
                control['derived_loop_sha256'] != admission['derived_loop_sha256'] or
                control['original_frozen_training_source_modified'] is not False):
            raise ValueError('The final curriculum must bind its actual authorized-prefix controller')
        self.control = control
        for key in ['entry', 'derived_loop']:
            path = Path(control[key])
            artifacts.add(path, {'sha256': control[key + '_sha256'], 'bytes': path.stat().st_size})
        contract = artifacts.json(self.root / 'contract.json')
        if (contract.get('authorized_prefix_control') != control or contract.get('code') != self.code or
                artifacts.json(contract['recipe']) != recipe or contract['recipe_sha256'] != digest(contract['recipe'])):
            raise ValueError('The actual resumed curriculum contract has a different prefix or execution source')
        for index, stage in enumerate(STAGES[:self.count]):
            imported = artifacts.json(self.root / stage / 'manifest.json')
            expected_kind = 'verified_imported_foundation_course' if index == 0 else 'user_boundary_selected_foundation_course'
            result = admission['courses'][index]
            if (imported.get('status') != 'complete' or imported.get('kind') != expected_kind or
                    imported.get('verification') != result or whole['verification']['courses'][index] != result or
                    imported.get('source_admission') != declaration['admission']['path'] or
                    imported.get('source_admission_sha256') != declaration['admission']['sha256'] or
                    imported.get('original_course_directory') != str(self.original_root / stage) or
                    imported.get('training_source_code') != result['training_code'] or
                    imported.get('independent_test_used') is not False):
                raise ValueError('An imported course must preserve its exact original result and prefix admission')
        self.prior = None
        if self.count == 2:
            first_path = self.original_root / STAGES[0] / 'manifest.json'
            if admission['original_course1_manifest_sha256'] != digest(first_path):
                raise ValueError('The original completed first-course proof changed')
            validate_direct_selection(admission['courses'][1], self.authorization)
        else:
            prior_declaration = pinned_json(artifacts, declaration['prior_prefix'])
            if prior_declaration.get('prefix_stages') != list(STAGES[:2]):
                raise ValueError('A third-course boundary must retain its exact earlier two-course authorization')
            prior_admission = pinned_json(artifacts, prior_declaration['admission'])
            prior_contract = artifacts.json(self.original_root / 'contract.json')
            # These are the preserved producer contract and imported results, not
            # an invented completed four-course manifest for the earlier run.
            prior_context = dict(code=prior_contract['code'], verification=dict(
                authorized_prefix_control=prior_contract['authorized_prefix_control'],
                courses=prior_admission['courses']))
            self.prior = AuthorizedPrefix(artifacts, declaration['prior_prefix']['path'],
                                          self.original_root, original, prior_context)
            if (admission.get('prior_two_course_admission') != prior_declaration['admission']['path'] or
                    admission.get('prior_two_course_admission_sha256') != prior_declaration['admission']['sha256'] or
                    admission['courses'][:2] != self.prior.admission['courses'] or self.prior.code != self.code):
                raise ValueError('The third-course boundary changed its actual earlier two-course prefix')
            validate_conditional_selection(admission['courses'][2], self.authorization)
        selected = admission['courses'][self.count - 1]
        if (admission['selected_checkpoint'] != selected['selected_checkpoint'] or
                admission['selected_checkpoint_sha256'] != selected['selected_checkpoint_sha256']):
            raise ValueError('The prefix parent differs from the actual user-selected checkpoint')

    @property
    def extra_whole_inputs(self):
        return {self.control[key] for key in ['admission', 'entry', 'derived_loop']}

    def course(self, index, parent):
        if self.prior and index < 2:
            return self.prior.course(index, parent)
        stage = STAGES[index]
        expected, candidates, code = course_contract(self.recipe, index, self.original_root, parent,
                                                     self.artifacts, self.code)
        result = self.admission['courses'][index]
        if index == 0:
            proof = self.artifacts.proof(self.original_root / stage / 'manifest.json', COURSE_KIND)
            if proof['verification'] != result:
                raise ValueError('Imported first course differs from its completed original proof')
        else:
            # Normalize the verified admission's bindings in memory. No ordinary
            # completed-course proof or successful original stopping rule is invented.
            name = result['selected_checkpoint']
            proof = dict(config=expected, code=self.code, verification=result,
                         inputs=self.admission['source_bound_files'],
                         outputs={name: self.admission['source_bound_files'][name]})
        return dict(proof=proof, expected=expected, candidates=candidates, training_code=code,
                    stage_root=self.original_root / stage, recipe=self.recipe,
                    recipe_path=self.declaration['source_recipe']['path'])

    def is_authorized_boundary(self, index):
        return 0 < index < self.count

    def validate_selection(self, index, result):
        if self.prior and index < 2:
            return self.prior.validate_selection(index, result)
        if index == self.count - 1:
            return (validate_direct_selection(result, self.authorization) if self.count == 2 else
                    validate_conditional_selection(result, self.authorization))
        from .curriculum_selection import validate_course_selection
        return validate_course_selection(result)

    def summary(self):
        result = dict(declaration=str(self.path), declaration_sha256=digest(self.path),
                    admission=self.declaration['admission'], source_recipe=self.declaration['source_recipe'],
                    authorization_binding=self.declaration['authorization'], authorization=self.authorization,
                    original_second_course_stopping_policy_met=False, prefix_stages=list(STAGES[:self.count]))
        if self.prior:
            result.update(prior_prefix_binding=self.declaration['prior_prefix'], prior_prefix=self.prior.summary(),
                          original_third_course_stopping_policy_met=False, original_third_course_minimum_steps_met=True)
        return result


def validate_handoff_authorization(proof):
    """Check the small explicit authorization graph again before a consumer starts SFT."""
    return _validate_authorization_graph(proof, proof['verification'].get('authorized_prefix', {}))


def _validate_authorization_graph(proof, value, *, require_two=False):
    path = Path(value.get('declaration', ''))
    binding = proof['inputs'].get(str(path), {})
    if digest(path) != value.get('declaration_sha256') or binding.get('sha256') != value.get('declaration_sha256'):
        raise ValueError('The handoff prefix declaration changed or is not input-bound')
    declaration = json.loads(path.read_text())
    stages = declaration.get('prefix_stages')
    if (declaration.get('kind') != DECLARATION_KIND or
            stages not in [list(STAGES[:2]), list(STAGES[:3])] or
            (require_two and stages != list(STAGES[:2]))):
        raise ValueError('The handoff requires its explicit ordered course-prefix authorization')
    for key, field in [('admission', 'admission'), ('source_recipe', 'source_recipe'), ('authorization', 'authorization_binding')]:
        declared = declaration[key]
        if (declared != value.get(field) or proof['inputs'].get(declared['path'], {}).get('sha256') != declared['sha256'] or
                digest(declared['path']) != declared['sha256']):
            raise ValueError('The handoff authorization graph is missing or changed')
    authorization = json.loads(Path(declaration['authorization']['path']).read_text())
    if (authorization != value.get('authorization') or value.get('prefix_stages') != stages or
            value.get('original_second_course_stopping_policy_met') is not False):
        raise ValueError('The handoff must preserve the actual user authorization and unmet original stopping policy')
    if stages == list(STAGES[:3]):
        prior = declaration.get('prior_prefix')
        if (prior != value.get('prior_prefix_binding') or not isinstance(prior, dict) or
                value.get('original_third_course_stopping_policy_met') is not False or
                value.get('original_third_course_minimum_steps_met') is not True or
                value.get('prior_prefix', {}).get('declaration') != prior.get('path') or
                value.get('prior_prefix', {}).get('declaration_sha256') != prior.get('sha256')):
            raise ValueError('The handoff lost its original two-course prefix or conditional third-course boundary')
        _validate_authorization_graph(proof, value['prior_prefix'], require_two=True)
        validate_conditional_selection(proof['verification']['courses'][2], authorization)
    return authorization
