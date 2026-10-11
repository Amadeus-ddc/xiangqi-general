import copy
import json
import sys

import pytest

from xqgeneral.evidence import atomic_json, position_key
from xqgeneral.search_distillation import SearchMiner
from test_search_distillation import ControlledOracle, ControlledPredictor, root_record

POLICY = {'do_sample': True, 'temperature': .6, 'top_k': 20, 'top_p': .95, 'max_new_tokens': 8}


class RetryingPredictor(ControlledPredictor):
    def __init__(self, failures=0, defect='missing', invalid_candidate=False, incorrect_child=False):
        super().__init__(incorrect_child=incorrect_child)
        self.failures, self.defect, self.invalid_candidate = failures, defect, invalid_candidate
        self.calls, self.root_calls, self.raws = [], {}, []

    def generate(self, row, question, budget, **options):
        self.calls.append({'id': row['id'], 'moves': list(row['moves']), 'options': options})
        value = json.loads(super().generate(row, question, budget))
        if not row['moves']:
            index = self.root_calls.get(row['id'], 0)
            self.root_calls[row['id']] = index + 1
            if index < self.failures:
                if self.defect == 'missing': value.pop('evaluation')
                elif self.defect == 'perspective': value['evaluation']['perspective'] = 'red'
                elif self.defect == 'bool': value['evaluation']['value'] = True
                elif self.defect == 'overflow': value['evaluation']['value'] = 2**4096
                elif self.defect == 'invalid_json':
                    raw = '原始格式错误，不是 JSON'
                    self.raws.append(raw)
                    return raw
            if self.invalid_candidate:
                value['candidates'] = ['a0a0']
        raw = json.dumps(value, ensure_ascii=False)
        self.raws.append(raw)
        return raw


class NeverOracle:
    def analyze(self, *args, **kwargs): raise AssertionError('Rejected roots must not reach engine search')


def miner(predictor, budget=3, oracle=None, **kwargs):
    return SearchMiner(predictor, oracle or ControlledOracle(), set(), generation_policy=POLICY,
        generation_seed=20260828, max_root_retries=budget, **kwargs)


@pytest.mark.parametrize('defect', ['missing', 'perspective', 'bool', 'overflow', 'invalid_json'])
def test_fourth_numerical_root_attempt_can_succeed_without_overwriting_failed_outputs(defect):
    predictor = RetryingPredictor(failures=3, defect=defect)
    search = miner(predictor)
    query, trace, reason = search.mine(root_record())
    assert query is not None and reason == 'accepted_for_consolidation'
    attempts = trace[0]['root_generation_attempts']
    assert len(attempts) == 4 and trace[0]['root_generation'] == attempts[-1]
    assert [a['root_evaluation_acceptance']['valid'] for a in attempts] == [False, False, False, True]
    assert [a['root_evaluation_acceptance']['attempt'] for a in attempts] == [1, 2, 3, 4]
    assert [a['raw'] for a in attempts] == predictor.raws[:4]
    # An integer can satisfy the structural schema and still overflow numeric conversion.
    assert all(a['verification']['valid'] == (defect == 'overflow') for a in attempts[:3])
    assert [a['generation']['call_index'] for a in attempts] == [0, 1, 2, 3]
    assert len({a['generation']['seed'] for a in attempts}) == 4
    assert {c['generation']['call_index'] for c in trace[0]['children']} == {4, 5, 6}
    assert search.counts['root_evaluation_retries'] == 3


@pytest.mark.parametrize('budget', [0, 1, 3])
def test_exhausted_numerical_retries_reject_before_oracle_search(budget):
    predictor = RetryingPredictor(failures=10)
    search = miner(predictor, budget, NeverOracle())
    query, trace, reason = search.mine(root_record())
    assert query is None and reason == 'invalid_root_evaluation'
    attempts = trace[0]['root_generation_attempts']
    assert len(attempts) == len(predictor.calls) == budget + 1
    assert all(a['root_evaluation_acceptance']['errors'] == ['invalid_evaluation_contract'] for a in attempts)
    assert search.counts['root_evaluation_retry_exhaustions'] == 1


def test_valid_evaluation_with_illegal_candidates_is_not_a_numerical_retry():
    predictor = RetryingPredictor(invalid_candidate=True)
    query, trace, reason = miner(predictor, oracle=NeverOracle()).mine(root_record())
    assert query is None and reason == 'invalid_root_candidates'
    assert len(predictor.calls) == len(trace[0]['root_generation_attempts']) == 1
    assert trace[0]['root_generation']['root_evaluation_acceptance']['valid']
    assert not trace[0]['root_generation']['verification']['valid']


def test_child_failures_recurse_without_using_the_root_retry_budget():
    predictor = RetryingPredictor(incorrect_child=True)
    search = miner(predictor, max_depth=0)
    query, trace, reason = search.mine(root_record())
    assert query is None and reason == 'recursion_limit'
    assert len(trace[0]['root_generation_attempts']) == 1
    assert len(predictor.calls) == 4 and search.counts['root_evaluation_retries'] == 0
    assert search.counts['recursive_descents'] == 1


def test_retries_and_child_seeds_are_identical_after_completed_prefix_skip_and_root_reordering():
    def run(rows):
        search = miner(RetryingPredictor(failures=2))
        return {row['id']: search.mine(row) for row in rows}
    rows = [dict(root_record(), id=identity) for identity in ['one', 'two', 'three']]
    continuous = run(rows)
    assert continuous == {**run(rows[:1]), **run(rows[1:])} == run(list(reversed(rows)))
    assert all(value[2] == 'accepted_for_consolidation' for value in continuous.values())


@pytest.mark.parametrize('budget', [-1, True, 1.5, '3'])
def test_invalid_retry_budget_rejected_before_any_generation(budget):
    predictor = RetryingPredictor()
    with pytest.raises(ValueError): miner(predictor, budget)
    assert predictor.calls == []


def test_positive_retries_require_sampling_and_reserved_root_is_rejected_before_decoding():
    predictor = RetryingPredictor()
    with pytest.raises(ValueError): SearchMiner(predictor, NeverOracle(), set(), max_root_retries=3)
    search = SearchMiner(predictor, NeverOracle(), {position_key(root_record()['fen'])},
        generation_policy=POLICY, max_root_retries=3)
    assert search.mine(root_record()) == (None, [], 'reserved_position')
    assert predictor.calls == []


def test_unspecified_retry_policy_retains_the_original_raw_trace():
    first = SearchMiner(ControlledPredictor(), ControlledOracle(), set()).mine(root_record())
    second = SearchMiner(ControlledPredictor(), ControlledOracle(), set(), max_root_retries=None).mine(root_record())
    assert first == second and 'root_generation_attempts' not in first[1][0]


def test_retry_budget_reaches_cli_and_changed_budget_cannot_resume_before_student_load(tmp_path, monkeypatch):
    from test_search_pilot import ControlledChain, setup_pilot
    from xqgeneral import search_distillation, search_pilot
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    policy = tmp_path/'sampling.json'; atomic_json(policy, POLICY)
    settings = json.loads(config.read_text());settings.update(student_generation_config=str(policy), max_root_retries=3)
    atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, query)
    output = tmp_path/'pilot'
    search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    command = next(c for c in chain.commands if 'xqgeneral.search_distillation' in c)
    assert command[command.index('--max-root-retries') + 1] == '3'
    mining = output/'mining';proof = json.loads((mining/'manifest.json').read_text())
    assert proof['config']['max_root_retries'] == proof['verification']['max_root_retries'] == 3
    (mining/'manifest.json').unlink()
    changed = copy.deepcopy(command[command.index('xqgeneral.search_distillation'):])
    changed[changed.index('--max-root-retries') + 1] = '2'
    monkeypatch.setattr(sys, 'argv', changed)
    with pytest.raises(ValueError, match='continuation inputs'): search_distillation.main()
    assert chain.student_loads == 1
