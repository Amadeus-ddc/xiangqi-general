"""Prompt routing controls; no real student/teacher quality claim."""
import hashlib
import json
import sys

import pytest

from test_search_distillation import ControlledOracle, ControlledPredictor, root_record
from test_search_pilot import ControlledChain, setup_pilot
from xqgeneral import search_distillation as search, search_pilot
from xqgeneral.evidence import atomic_json, position_key
from xqgeneral.explanations import EXPLANATION_QUESTION


PROFILE = 'numeric_six_ply_v1'
POLICY = {'do_sample': True, 'temperature': .6, 'top_k': 20, 'top_p': .95, 'max_new_tokens': 4096}


class CapturingPredictor(ControlledPredictor):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, row, question, budget, **options):
        self.calls.append({'record': row, 'question': question, 'budget': budget, 'options': options})
        raw = super().generate(row, question, budget)
        if question == search.NUMERIC_CHILD_QUESTION:
            value = json.loads(raw)
            return json.dumps({key: value[key] for key in ['move', 'pv', 'candidates', 'evaluation']})
        return raw


def make_search(predictor, acceptance='strict', sampled=True, **kwargs):
    return search.SearchMiner(predictor, ControlledOracle(), set(), child_contract='move_eval',
        max_root_retries=0, generation_policy=POLICY if sampled else None,
        generation_seed=20261013, acceptance_policy=acceptance,
        root_candidate_policy='paper_fill' if acceptance == 'paper_recursive' else 'strict', **kwargs)


@pytest.mark.parametrize('profile', ['unknown', 'numeric_six_ply_v0', None, 3])
def test_unknown_child_prompt_profile_is_rejected(profile):
    with pytest.raises(ValueError, match='child prompt'):
        make_search(CapturingPredictor(), child_prompt_profile=profile)


def test_numerical_prompt_cannot_claim_the_full_child_contract():
    with pytest.raises(ValueError, match='move_eval'):
        search.SearchMiner(None, None, set(), child_prompt_profile=PROFILE)


@pytest.mark.parametrize('acceptance', ['strict', 'paper_recursive'])
@pytest.mark.parametrize('sampled', [False, True])
def test_only_children_receive_short_prompt_with_original_sampling_and_budget(acceptance, sampled):
    predictor = CapturingPredictor()
    miner = make_search(predictor, acceptance, sampled, child_prompt_profile=PROFILE)
    query, trace, reason = miner.mine(root_record())
    assert query is not None and reason == 'accepted_for_consolidation'
    assert predictor.calls[0]['question'] == EXPLANATION_QUESTION
    assert all(call['question'] == search.NUMERIC_CHILD_QUESTION for call in predictor.calls[1:])
    assert all(call['record']['question'] == EXPLANATION_QUESTION for call in predictor.calls)
    assert all(call['budget'] == (4096 if sampled else 768) for call in predictor.calls)
    assert '六个半步' in search.NUMERIC_CHILD_QUESTION and '三个候选' in search.NUMERIC_CHILD_QUESTION
    assert hashlib.sha256(search.NUMERIC_CHILD_QUESTION.encode()).hexdigest() == '76adba2c8876936be2258f5ab921447d73b6a6ca0997d39e86316af6aef8cfd0'
    assert 'prompt' not in trace[0]['root_generation']
    for child in trace[0]['children']:
        assert child['prompt'] == {'profile': PROFILE, 'question': search.NUMERIC_CHILD_QUESTION,
            'question_sha256': hashlib.sha256(search.NUMERIC_CHILD_QUESTION.encode()).hexdigest()}
        assert json.loads(child['raw']) == child['analysis']
        assert set(child['analysis']) == {'move', 'pv', 'candidates', 'evaluation'}
        assert child['verification']['valid'] is False and child['child_acceptance']['valid'] is True
    if sampled:
        generations = [trace[0]['root_generation']['generation'],
                       *[child['generation'] for child in trace[0]['children']]]
        assert sorted(item['call_index'] for item in generations) == list(range(4))
        assert all(item['settings'] == POLICY for item in generations)
        assert all(call['options']['sampling'] == {k: v for k, v in POLICY.items() if k != 'max_new_tokens'}
                   for call in predictor.calls)
    else:
        assert all(not call['options'] for call in predictor.calls)


@pytest.mark.parametrize('acceptance', ['strict', 'paper_recursive'])
def test_legacy_default_preserves_complete_raw_trace_and_generation_seeds(acceptance):
    implicit, explicit, numeric = [CapturingPredictor() for _ in range(3)]
    first = make_search(implicit, acceptance).mine(root_record())
    second = make_search(explicit, acceptance, child_prompt_profile='full_explanation').mine(root_record())
    third = make_search(numeric, acceptance, child_prompt_profile=PROFILE).mine(root_record())
    assert first == second and implicit.calls == explicit.calls
    assert all(call['question'] == EXPLANATION_QUESTION for call in implicit.calls)
    assert first[0]['target_fields'] == third[0]['target_fields']
    assert first[2] == third[2] == 'accepted_for_consolidation'
    assert [call['options'] for call in implicit.calls] == [call['options'] for call in numeric.calls]


def test_cached_numerical_child_keeps_actual_prompt_without_new_generation():
    predictor = CapturingPredictor()
    miner = make_search(predictor, child_prompt_profile=PROFILE, reuse_child_analysis=True)
    row = search.descend(root_record(), 'h0g2')
    generated = search.child_model_analysis(miner, row)
    cached = dict(generated, record=row, child_acceptance=miner.child_acceptance(row, generated))
    before = len(predictor.calls)
    reused, attempts = miner.root_analysis(row, cached)
    assert len(predictor.calls) == before
    assert reused['prompt'] == generated['prompt']
    assert reused['generation'] == generated['generation']
    assert reused['raw'] == generated['raw'] and reused['root_evaluation_acceptance']['attempt'] == 0
    assert attempts == [reused]


def test_short_child_raw_continuation_still_cannot_enter_reserved_position():
    miner = make_search(CapturingPredictor(), 'paper_recursive', child_prompt_profile=PROFILE)
    child = search.descend(root_record(), 'h0g2')
    reserved = search.descend(child, 'b9c7')
    miner.reserved.add(position_key(reserved['fen']))
    query, trace, reason = miner.mine(root_record())
    assert query is None and reason == 'reserved_child_continuation'
    assert trace[0]['children'][0]['prompt']['profile'] == PROFILE


@pytest.mark.parametrize('acceptance', ['strict', 'paper_recursive'])
def test_numeric_prompt_does_not_bypass_engine_evaluation_error_gate(acceptance):
    class WrongScore(CapturingPredictor):
        def generate(self, row, question, budget, **options):
            value = json.loads(super().generate(row, question, budget, **options))
            if row['moves']:
                value['evaluation']['value'] = 999999
            return json.dumps(value)

    miner = make_search(WrongScore(), acceptance, child_prompt_profile=PROFILE, max_depth=0)
    query, trace, reason = miner.mine(root_record())
    assert query is None and reason == 'recursion_limit'
    assert all(child['analysis']['evaluation']['value'] == 999999 for child in trace[0]['children'])
    assert all(json.loads(child['raw']) == child['analysis'] for child in trace[0]['children'])


def test_pilot_forwards_profile_and_changed_resume_is_rejected(tmp_path, monkeypatch):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    saved = json.loads(config.read_text());saved['child_prompt_profile'] = PROFILE
    atomic_json(config, saved)
    chain = ControlledChain(monkeypatch, query)
    output = tmp_path / 'pilot'
    search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    command = next(value for value in chain.commands if 'xqgeneral.search_distillation' in value)
    assert command[command.index('--child-prompt-profile') + 1] == PROFILE
    proof = json.loads((output / 'mining/manifest.json').read_text())
    assert proof['config']['child_prompt_profile'] == proof['verification']['child_prompt']['profile'] == PROFILE
    assert proof['verification']['child_prompt']['question'] == search.NUMERIC_CHILD_QUESTION
    (output / 'mining/manifest.json').unlink()
    changed = command[command.index('xqgeneral.search_distillation'):]
    changed[changed.index('--child-prompt-profile') + 1] = 'full_explanation'
    monkeypatch.setattr(sys, 'argv', changed)
    with pytest.raises(ValueError, match='continuation inputs'):
        search.main()
    assert chain.student_loads == 1


@pytest.mark.parametrize('profile,contract', [('unknown', 'move_eval'), (PROFILE, 'full')])
def test_invalid_pilot_prompt_fails_before_output_or_model_load(tmp_path, monkeypatch, profile, contract):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    saved = json.loads(config.read_text());saved.update(child_prompt_profile=profile, child_contract=contract)
    atomic_json(config, saved)
    chain = ControlledChain(monkeypatch, query)
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError, match='child prompt|move_eval'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    assert not output.exists() and not chain.commands
    assert chain.student_loads == chain.teacher_loads == 0


def test_cli_rejects_short_prompt_full_contract_before_reading_missing_assets(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['search', '--checkpoint', 'missing.pt', '--output', 'missing-output',
                                     '--child-prompt-profile', PROFILE])
    with pytest.raises(ValueError, match='move_eval'):
        search.main()
