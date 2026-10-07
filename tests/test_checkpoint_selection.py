import json
import pytest
import torch

from xqgeneral.evidence import atomic_json, digest, manifest
from xqgeneral.select_checkpoint import functional_score, qa_selection_accuracy, read_validation, snapshot_checkpoint


def test_atomic_checkpoint_snapshot_omits_optimizer_and_survives_source_replacement(tmp_path):
    source = tmp_path/'latest.pt'
    checkpoint = {'step':2000, 'trainable':{'bridge':torch.ones(2)}, 'config':{'mode':'bridge'},
                  'input_hashes':{'data':'pinned'}, 'code':{'revision':'source'},
                  'optimizer':{'large_state':torch.zeros(100)}}
    torch.save(checkpoint, source)
    root, step = snapshot_checkpoint(source, tmp_path/'candidates')
    assert step == 2000
    kept = torch.load(root/'adapter.pt', weights_only=True)
    assert 'optimizer' not in kept and torch.equal(kept['trainable']['bridge'], torch.ones(2))
    newer = dict(checkpoint, step=4000, trainable={'bridge':torch.zeros(2)})
    partial = tmp_path/'partial.pt';torch.save(newer, partial);partial.replace(source)
    archived = torch.load(root/'source-checkpoint.pt', weights_only=True)
    assert archived['step'] == 2000 and torch.equal(archived['trainable']['bridge'], torch.ones(2))
    # A best-NLL adapter can have the same step and tensors as a latest checkpoint.
    adapter = tmp_path/'adapter.pt';torch.save(kept, adapter)
    assert snapshot_checkpoint(adapter, tmp_path/'candidates') == (root, 2000)
    altered = dict(kept, trainable={'bridge':torch.zeros(2)})
    torch.save(altered, adapter)
    with pytest.raises(ValueError, match='different source'):
        snapshot_checkpoint(adapter, tmp_path/'candidates')


def test_selection_rejects_test_or_constrained_results_and_tampered_evidence(tmp_path):
    common = {'split':'validation', 'raw_generation':True, 'oracle_repairs':0, 'examples':96}
    move, plan = {**common, 'no_mistake_rate':.8}, {**common, 'contract_valid_rate':.4}
    assert functional_score(move, plan) > 0
    for bad in [dict(move, split='test'), dict(move, legality_is_imposed_by_decoding=True),
                dict(move, oracle_repairs=1), dict(move, examples=0), dict(move, no_mistake_rate=float('nan'))]:
        with pytest.raises(ValueError):
            functional_score(bad, plan)
    checkpoint = tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'pinned checkpoint')
    root = tmp_path/'evaluation';atomic_json(root/'metrics.json', move)
    atomic_json(root/'manifest.json', manifest('test_fixture', {}, [checkpoint], [root/'metrics.json'], move))
    assert read_validation(root, checkpoint) == move
    atomic_json(root/'metrics.json', dict(move, no_mistake_rate=1.0))
    with pytest.raises(ValueError, match='output changed'):
        read_validation(root, checkpoint)


def test_explanation_selection_can_prefer_complete_answers_over_move_only_quality():
    common = {'split': 'validation', 'raw_generation': True, 'oracle_repairs': 0, 'examples': 96}
    move_only = {**common, 'no_mistake_rate': .9}
    balanced = {**common, 'no_mistake_rate': .8}
    plan = {**common, 'contract_valid_rate': .8}
    weak = {**common, 'structured_contract_valid_rate': 0.0}
    complete = {**common, 'structured_contract_valid_rate': .9}
    assert functional_score(move_only, plan) > functional_score(balanced, plan)
    assert functional_score(move_only, plan, weak) < functional_score(balanced, plan, complete)
    assert functional_score(balanced, plan, complete) == pytest.approx(.84)
    for bad in [dict(complete, split='test'), dict(complete, raw_generation=False),
                dict(complete, rule_legal_constraints=True), dict(complete, oracle_repairs=1),
                dict(complete, structured_contract_valid_rate=float('nan'))]:
        with pytest.raises(ValueError):
            functional_score(balanced, plan, bad)


def test_foundation_selection_checks_balanced_qa_evidence_and_retains_move_quality():
    common = {'split': 'validation', 'raw_generation': True, 'oracle_repairs': 0, 'examples': 96}
    move = {**common, 'no_mistake_rate': .8}
    plan = {**common, 'contract_valid_rate': .5}
    groups = {f'{stage}/{task}': {'n': 12, 'accuracy': .9}
              for stage in ['static_current', 'static_future']
              for task in ['piece', 'count', 'locate', 'empty', 'material', 'rank']}
    groups.update({f'{stage}/{task}': {'n': 12, 'accuracy': .9}
                   for stage in ['dynamic_current', 'dynamic_future']
                   for task in ['legal', 'illegal', 'moves', 'captures', 'checks']})
    qa = {'split': 'validation', 'raw_generation': True, 'oracle_used': False,
          'examples': 264, 'accuracy': .9, 'by_task': groups}
    assert qa_selection_accuracy(qa) == pytest.approx(.9)
    assert functional_score(move, plan, qa=qa) == pytest.approx(.78)
    weaker_qa = dict(qa, accuracy=.5, by_task={k: dict(v, accuracy=.5) for k, v in groups.items()})
    assert functional_score(move, plan, qa=qa) > functional_score(move, plan, qa=weaker_qa)
    for bad in [dict(qa, split='test'), dict(qa, oracle_used=True), dict(qa, raw_generation=False),
                dict(qa, rule_legal_constraints=True), dict(qa, examples=0), dict(qa, accuracy=float('nan')),
                dict(qa, by_task={k: v for k, v in groups.items() if not k.startswith('dynamic_future/')}),
                dict(qa, by_task={**groups, 'static_current/piece': {'n': 11, 'accuracy': .9}}),
                dict(qa, accuracy=.99)]:
        with pytest.raises(ValueError):
            functional_score(move, plan, qa=bad)
    with pytest.raises(ValueError, match='either'):
        functional_score(move, plan, {**common, 'structured_contract_valid_rate': .8}, qa)
