"""Reopen real native CPU collection fixtures with the optional mate policy."""
import json
from pathlib import Path
import sys

import pytest

from test_merge_search import rebind, setup_merge
from xqgeneral import collect_search, distillation_preflight, merge_search
from xqgeneral.evidence import atomic_json, load_jsonl, write_jsonl
from xqgeneral.explanations import IMMEDIATE_MATE_CONSISTENCY_PROFILE
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.rules import adjudicate


def recollect_with_mate_policy(leaf, monkeypatch):
    config = json.loads((leaf / 'manifest.json').read_text())['config']
    output = leaf.parent / 'mate-filtered-labels'
    arguments = ['collect_search']
    for name, value in config.items():
        if name != 'output' and value is not None:
            arguments += ['--' + name.replace('_', '-'), str(value)]
    arguments += ['--output', str(output), '--require-immediate-mate-consistency']
    monkeypatch.setattr(sys, 'argv', arguments)
    collect_search.main()
    return output


@pytest.mark.parametrize('strict_count', [1, 2])
def test_merger_reconstructs_each_source_policy_and_reports_mixed_sources_honestly(
        tmp_path, monkeypatch, strict_count):
    parent, legacy, _ = setup_merge(tmp_path, monkeypatch)
    leaves = [recollect_with_mate_policy(leaf, monkeypatch) if i < strict_count else leaf
              for i, leaf in enumerate(legacy)]
    original = merge_search.rebuild_labels
    observed = []

    def record_policy(*args, **kwargs):
        observed.append(kwargs.get('require_immediate_mate_consistency', False))
        return original(*args, **kwargs)

    monkeypatch.setattr(merge_search, 'rebuild_labels', record_policy)
    output = tmp_path / 'merged'
    summary = merge_search.merge_collections(leaves, parent, output)
    assert observed == [i < strict_count for i in range(2)]
    assert summary['immediate_mate_consistency_source_collections'] == strict_count
    assert summary['all_source_collections_require_immediate_mate_consistency'] is (strict_count == 2)
    assert summary['immediate_mate_consistency_profile'] == IMMEDIATE_MATE_CONSISTENCY_PROFILE
    assert summary['all_strategic_prose_semantics_verified'] is False
    assert summary['accepted_training_roots'] == 35
    for split in ['validation', 'test']:
        assert (output / (split + '.jsonl')).read_bytes() == (legacy[0] / (split + '.jsonl')).read_bytes()
    reopened = distillation_preflight.collected_dataset(output, parent, BoundInputs())
    assert reopened['verification'] == summary
    assert observed == [i < strict_count for i in range(2)] * 2


@pytest.mark.parametrize('mutation', ['disabled_config', 'unknown_profile', 'false_verification'])
def test_collection_cannot_claim_a_different_mate_policy(tmp_path, monkeypatch, mutation):
    parent, leaves, _ = setup_merge(tmp_path, monkeypatch)
    leaf = recollect_with_mate_policy(leaves[0], monkeypatch)
    path = leaf / 'manifest.json'
    proof = json.loads(path.read_text())
    if mutation == 'disabled_config':
        proof['config']['require_immediate_mate_consistency'] = False
    elif mutation == 'unknown_profile':
        proof['verification']['immediate_mate_consistency_profile'] = 'unrelated'
    else:
        proof['verification']['immediate_mate_score_consistency_required'] = False
    atomic_json(path, proof)
    with pytest.raises(ValueError, match='policy differs'):
        merge_search.merge_collections([leaf, leaves[1]], parent, tmp_path / 'rejected')


def test_claimed_filtered_leaf_is_natively_rechecked_even_with_rebound_output_hash(
        tmp_path, monkeypatch):
    parent, leaves, _ = setup_merge(tmp_path, monkeypatch)
    leaf = recollect_with_mate_policy(leaves[0], monkeypatch)
    train = leaf / 'train.jsonl'
    rows = load_jsonl(train)
    first = rows[0]
    value = json.loads(first['answer'])
    assert not adjudicate(first['initial_fen'], first['moves'] + value['pv'][:1])['ended']
    value['evaluation'] = {'type': 'mate', 'value': 1, 'perspective': 'side_to_move'}
    first['answer'] = json.dumps(value, ensure_ascii=False)
    write_jsonl(train, rows)
    rebind(leaf / 'manifest.json', 'outputs', train)
    with pytest.raises(ValueError, match='retains an inconsistent PV'):
        merge_search.merge_collections([leaf, leaves[1]], parent, tmp_path / 'rejected')
