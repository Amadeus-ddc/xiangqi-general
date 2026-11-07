"""Native CLI orchestration; descendant admission is tested separately with toy states."""
import json
from pathlib import Path

import pytest

from test_search_pilot import setup_pilot, ControlledChain
from test_search_protection_pipeline import certificate
from xqgeneral import search_parent, search_pilot
from xqgeneral.evidence import atomic_json, digest


@pytest.mark.parametrize('parent_family', ['clean', 'auxiliary', 'distilled'])
def test_pilot_forwards_protection_and_reports_actual_stage_step_and_unchanged_counts(tmp_path, monkeypatch, parent_family):
    pipeline, validation, config, query, capability = setup_pilot(tmp_path, monkeypatch)
    settings = json.loads(config.read_text()); pool = Path(settings['recorded_inputs'])
    positions = set(json.loads((pool / 'heldout-positions.json').read_text()))
    cert, reader, calls = certificate(tmp_path, monkeypatch, positions, len(positions))
    settings['protection_manifest'] = str(cert); atomic_json(config, settings)
    descendant = parent_family != 'clean'
    auxiliary = parent_family == 'auxiliary'
    if descendant:
        evaluation = json.loads(Path(settings['evaluation_config']).read_text())
        token_path = Path(evaluation['token_preflight']); token = json.loads(token_path.read_text())
        token['kind'] = search_parent.AUXILIARY_TOKEN_KIND if auxiliary else search_parent.TOKEN_KIND
        atomic_json(token_path, token)
        # Native descendant admission and data validation are exercised in
        # test_search_parent; this seam controls only orchestration here.
        monkeypatch.setattr(search_pilot, 'validation_spec', lambda value: None)
        parent = json.loads((pipeline / 'manifest.json').read_text())
        parent.update(kind=search_parent.AUXILIARY_ROUND_KIND if auxiliary else search_parent.ROUND_KIND)
        parent['verification'] = dict(selected_step=17, trained_auxiliary_policy_rounds=1, trained_search_distillation_rounds=2)
        monkeypatch.setattr(search_pilot, 'checked_sft', lambda *args: (pipeline / 'sft/adapter.pt', parent))
        capability['kind'] = ('auxiliary_policy_capability_validation' if auxiliary else 'search_distillation_capability_validation')
        capability['verification'].update(actual_clean_initial_sft_completed_before_model_or_reviewer_load=False,
            actual_full_decoder_auxiliary_round_completed_before_evaluation=auxiliary,
            actual_full_decoder_distillation_round_completed_before_evaluation=not auxiliary,
            trained_auxiliary_policy_rounds=1, trained_search_distillation_rounds=2)
        atomic_json(validation / 'manifest.json', capability)
        def admit(source, parent, token, path, proof, evaluation, settings, pool_path, bound):
            _, note = reader(settings['protection_manifest'], bound, base_data=settings['data'], source_pool=pool_path)
            return dict(controlled_admission_seam=True, search_protection=note)
        monkeypatch.setattr(search_pilot, 'admit_descendant', admit)
    controlled = ControlledChain(monkeypatch, query)
    output = tmp_path / 'pilot'; result = search_pilot.run_pilot(pipeline, validation, 'controlled', config, output)
    command = next(command for command in controlled.commands if 'xqgeneral.search_distillation' in command)
    assert command[command.index('--protection-manifest') + 1] == str(cert)
    mining = json.loads((output / 'mining/manifest.json').read_text())
    assert mining['inputs'][str(cert)]['sha256'] == digest(cert)
    assert mining['verification']['search_protection']['protected_positions'] == len(positions)
    assert len(calls) >= 2
    assert result['new_SGD_rounds_from_this_pilot'] == 0
    assert result['actual_selected_step'] == (17 if descendant else 9)
    assert result['trained_auxiliary_policy_rounds'] == (1 if descendant else 0)
    assert result['trained_search_distillation_rounds'] == (2 if descendant else 0)
    assert result['clean_sft_and_paired_validation_completed_before_search'] is (not descendant)
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['kind'] == parent_family + '_recorded_search_yield_pilot'
