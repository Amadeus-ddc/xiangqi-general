"""Verify an explicitly authorized two-course prefix without changing its stopping history."""
from pathlib import Path
import json

from .curriculum_data import STAGES
from .curriculum_selection import COURSE_KIND, CURRENT_COURSE_METRIC, course_contract, selection_decision
from .evidence import digest
from .foundation_preflight import clean_recipe


DECLARATION_KIND = 'explicit_user_authorized_course_prefix'
DIRECT_BOUNDARY = 'explicit_user_authorized_direct_course_boundary'


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


class AuthorizedPrefix:
    """Resolve original prefix artifacts; later courses still use their complete ordinary proofs."""
    def __init__(self, artifacts, declaration_path, root, recipe, whole):
        self.artifacts = artifacts
        self.path = Path(declaration_path)
        self.declaration = declaration = artifacts.json(self.path)
        if (set(declaration) != {'kind', 'prefix_stages', 'admission', 'source_recipe', 'authorization'} or
                declaration['kind'] != DECLARATION_KIND or declaration['prefix_stages'] != list(STAGES[:2])):
            raise ValueError('Opt in to an exact two-course prefix declaration')
        self.admission = admission = pinned_json(artifacts, declaration['admission'])
        self.recipe = original = pinned_json(artifacts, declaration['source_recipe'])
        self.authorization = pinned_json(artifacts, declaration['authorization'])
        self.original_root = Path(admission['original_root'])
        self.root = Path(root)
        self.code = whole['code']
        clean_recipe(original)
        old_minimum = original['course_budgets'][1]['minimum_steps']
        stopping_claim = admission.get('original_stopping_policy_claimed_met',
                                      admission.get(f'old_{old_minimum}_stopping_requirement_claimed_met'))
        if (admission.get('status') != 'complete_explicit_user_boundary_two_course_prefix' or
                Path(admission['new_root']).resolve() != self.root.resolve() or
                self.original_root.resolve() == self.root.resolve() or
                Path(original['output']).resolve() != self.original_root.resolve() or
                dict(original, output=recipe['output']) != recipe or len(admission.get('courses', [])) != 2 or
                admission.get('source_training_code') != self.code or
                admission.get('original_recipe_or_sources_hot_edited') is not False or
                stopping_claim is not False or
                admission.get('independent_test_used') is not False or
                admission.get('authorization_sha256') != declaration['authorization']['sha256']):
            raise ValueError('Authorized prefix must bind the preserved recipe, source, authorization and actual two courses')
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
        for index, stage in enumerate(STAGES[:2]):
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
        first_path = self.original_root / STAGES[0] / 'manifest.json'
        if admission['original_course1_manifest_sha256'] != digest(first_path):
            raise ValueError('The original completed first-course proof changed')
        validate_direct_selection(admission['courses'][1], self.authorization)
        selected = admission['courses'][1]
        if (admission['selected_checkpoint'] != selected['selected_checkpoint'] or
                admission['selected_checkpoint_sha256'] != selected['selected_checkpoint_sha256']):
            raise ValueError('The prefix parent differs from the actual user-selected checkpoint')

    @property
    def extra_whole_inputs(self):
        return {self.control[key] for key in ['admission', 'entry', 'derived_loop']}

    def course(self, index, parent):
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

    def summary(self):
        return dict(declaration=str(self.path), declaration_sha256=digest(self.path),
                    admission=self.declaration['admission'], source_recipe=self.declaration['source_recipe'],
                    authorization_binding=self.declaration['authorization'], authorization=self.authorization,
                    original_second_course_stopping_policy_met=False, prefix_stages=list(STAGES[:2]))


def validate_handoff_authorization(proof):
    """Check the small explicit authorization graph again before a consumer starts SFT."""
    value = proof['verification'].get('authorized_prefix', {})
    path = Path(value.get('declaration', ''))
    binding = proof['inputs'].get(str(path), {})
    if digest(path) != value.get('declaration_sha256') or binding.get('sha256') != value.get('declaration_sha256'):
        raise ValueError('The handoff prefix declaration changed or is not input-bound')
    declaration = json.loads(path.read_text())
    if declaration.get('kind') != DECLARATION_KIND or declaration.get('prefix_stages') != list(STAGES[:2]):
        raise ValueError('The handoff requires its explicit ordered two-course prefix')
    for key, field in [('admission', 'admission'), ('source_recipe', 'source_recipe'), ('authorization', 'authorization_binding')]:
        declared = declaration[key]
        if (declared != value.get(field) or proof['inputs'].get(declared['path'], {}).get('sha256') != declared['sha256'] or
                digest(declared['path']) != declared['sha256']):
            raise ValueError('The handoff authorization graph is missing or changed')
    authorization = json.loads(Path(declaration['authorization']['path']).read_text())
    if (authorization != value.get('authorization') or value.get('prefix_stages') != list(STAGES[:2]) or
            value.get('original_second_course_stopping_policy_met') is not False):
        raise ValueError('The handoff must preserve the actual user authorization and unmet original stopping policy')
    return authorization
