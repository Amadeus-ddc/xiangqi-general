"""Real native terminals and adversarial proof readback, without model calls."""
from copy import deepcopy
import json

import pytest

from xqgeneral import terminal_audit as audit
from xqgeneral.rules import START_FEN, adjudicate, play, replay

NEGATIVE = '9/9/1N2k4/7R1/4R1b2/9/9/9/5K3/9 b - - 0 1'
TWO_REPLIES = '9/9/4k4/7R1/4R1b2/9/9/9/4K4/9 b - - 0 1'
POSITIVE = '4k4/9/5r3/9/9/9/3n5/5A3/5K3/9 b - - 0 1'
STALEMATE = '4k4/3R5/9/5R3/9/9/9/9/9/5K3 w - - 0 1'


def record(fen):
    return {'initial_fen': fen, 'moves': [], 'fen': fen, 'split': 'train'}


def branch(move, value):
    return {'move': move, 'pv': [move],
            'evaluation': {'type': 'mate', 'value': value, 'perspective': 'side_to_move'}}


@pytest.mark.parametrize('fen,move,value,winner,offset', [
    (NEGATIVE, 'e7d7', -3, 'red', 0),
    (POSITIVE, 'f7f2', 3, 'black', 1),
    (STALEMATE, 'f6f7', 3, 'red', 1),
])
def test_real_shorter_win_contradicts_a_candidate_estimate(fen, move, value, winner, offset):
    r, b = record(fen), branch(move, value)
    before = deepcopy((r, b))
    result = audit.audit_mate_branch(r, b)
    assert result['status'] == 'positive_distance_contradiction'
    assert result['certificate']['winner'] == winner
    assert result['fixed_winner_move_offset'] == offset
    assert audit.verify_mate_contradiction(r, b, result) == 1
    assert not result['original_score_certified_correct']
    assert not result['certificate']['shortest_mate_distance_proven']
    assert (r, b) == before
    if fen == STALEMATE:
        assert result['certificate']['proof']['terminal_kind'] == 'stalemate_loss'


def test_certificate_enumerates_both_opponent_replies_and_rechecks_them():
    r = record(TWO_REPLIES)
    result = audit.prove_terminal_win(r, 'red', max_winner_moves=1)
    assert result['status'] == 'proved'
    assert result['proof']['kind'] == 'all'
    assert set(result['proof']['replies']) == {'e7d7', 'e7f7'}
    assert len(result['proof']['children']) == 2
    assert audit.verify_terminal_certificate(r, result) == 1
    for name in ['missing', 'duplicate', 'wrong_winner', 'wrong_distance', 'wrong_terminal', 'illegal_choice']:
        bad = deepcopy(result)
        node = bad['proof']
        if name == 'missing':
            node['replies'].pop(); node['children'].pop()
        elif name == 'duplicate':
            node['children'][1] = deepcopy(node['children'][0])
        elif name == 'wrong_winner':
            bad['winner'] = 'black'
        elif name == 'wrong_distance':
            node['upper'] = 2
        elif name == 'wrong_terminal':
            node['children'][0]['proof']['child']['fen'] = START_FEN
        else:
            node['children'][0]['proof']['move'] = 'a0a9'
        with pytest.raises(ValueError):
            audit.verify_terminal_certificate(r, bad)


def test_history_and_candidate_bindings_reject_a_different_record_or_score():
    r, b = record(NEGATIVE), branch('e7d7', -3)
    result = audit.audit_mate_branch(r, b)
    changed = deepcopy(b)
    changed['evaluation']['value'] = -4
    with pytest.raises(ValueError, match='binding'):
        audit.verify_mate_contradiction(r, changed, result)
    cert = result['certificate']
    after = record(play(NEGATIVE, 'e7d7'))
    with pytest.raises(ValueError, match='history binding'):
        audit.verify_terminal_certificate(after, cert)


@pytest.mark.parametrize('field,value', [('schema_version', True), ('winner_moves_upper_bound', True)])
def test_certificate_numeric_fields_do_not_accept_boolean_aliases(field, value):
    r = record(TWO_REPLIES)
    result = audit.prove_terminal_win(r, 'red', max_winner_moves=1)
    result[field] = value
    with pytest.raises(ValueError):
        audit.verify_terminal_certificate(r, result)


def test_non_witness_and_resource_limit_do_not_validate_the_claimed_score():
    r, b = record(START_FEN), branch('a3a4', 3)
    bounded = audit.audit_mate_branch(r, b, max_winner_moves=1)
    assert bounded['status'] == 'no_witness_within_bound'
    assert bounded['proved_winner_moves_upper_bound'] is None
    limited = audit.audit_mate_branch(r, b, max_state_checks=1)
    assert limited['status'] == 'resource_exhausted'
    for result in [bounded, limited]:
        assert not result['positive_distance_contradiction']
        assert not result['original_score_certified_correct']
        with pytest.raises(ValueError):
            audit.verify_mate_contradiction(r, b, result)


def test_deadline_exhaustion_is_separate_from_a_bounded_non_witness(monkeypatch):
    clock = iter(range(100))
    monkeypatch.setattr(audit.time, 'monotonic', lambda: next(clock))
    result = audit.prove_terminal_win(record(TWO_REPLIES), 'red', seconds=0.1)
    assert result['status'] == 'resource_exhausted' and result['proof'] is None


def test_zero_move_bound_can_certify_an_already_realized_terminal_only():
    r = record(POSITIVE)
    r.update(moves=['f7f2'], fen=play(POSITIVE, 'f7f2'))
    proof = audit.prove_terminal_win(r, 'black', max_winner_moves=0)
    assert audit.verify_terminal_certificate(r, proof) == 0
    assert audit.prove_terminal_win(record(START_FEN), 'red', max_winner_moves=0)['status'] == 'no_witness_within_bound'


def test_full_history_cannot_continue_after_an_AXF_terminal():
    moves = ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2
    assert adjudicate(START_FEN, moves)['ended']
    moves.append('b0c2')
    r = record(START_FEN)
    r.update(moves=moves, fen=replay(START_FEN, moves)[-1])
    with pytest.raises(ValueError, match='beyond a terminal'):
        audit.prove_terminal_win(r, 'red')


@pytest.mark.parametrize('kwargs', [
    {'max_winner_moves': True}, {'max_winner_moves': -1}, {'max_winner_moves': 9},
    {'max_state_checks': True}, {'max_state_checks': 0},
    {'seconds': True}, {'seconds': 0}, {'seconds': float('nan')}, {'seconds': float('inf')},
])
def test_invalid_limits_fail_before_search(kwargs):
    with pytest.raises(ValueError):
        audit.prove_terminal_win(record(NEGATIVE), 'red', **kwargs)


@pytest.mark.parametrize('split', ['validation', 'test', None])
def test_candidate_training_audit_rejects_heldout_or_unspecified_split(split):
    r = record(NEGATIVE)
    r['split'] = split
    with pytest.raises(ValueError, match='explicit train'):
        audit.audit_mate_branch(r, branch('e7d7', -3))


@pytest.mark.parametrize('value', [True, 1, -1, 0, 2.5, '3'])
def test_candidate_score_requires_a_longer_integer_mate(value):
    with pytest.raises(ValueError):
        audit.audit_mate_branch(record(NEGATIVE), branch('e7d7', value))


def test_cli_reads_back_a_portable_manifest_and_preserves_its_inputs(tmp_path):
    source = tmp_path / 'case.json'
    source.write_text(json.dumps({'record': record(NEGATIVE), 'branch': branch('e7d7', -3)}))
    before = source.read_bytes()
    output = tmp_path / 'output'
    audit.main(['--input', str(source), '--output', str(output)])
    result = json.loads((output / 'audit.json').read_text())
    evidence = json.loads((output / 'manifest.json').read_text())
    assert audit.verify_mate_contradiction(record(NEGATIVE), branch('e7d7', -3), result) == 1
    assert evidence['inputs'][str(source)]['sha256'] == audit.digest(source)
    assert evidence['outputs'][str(output / 'audit.json')]['sha256'] == audit.digest(output / 'audit.json')
    assert evidence['verification']['model_or_engine_calls'] == 0
    assert evidence['verification']['new_SGD_updates'] == 0
    assert source.read_bytes() == before
    saved = (output / 'audit.json').read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        audit.main(['--input', str(source), '--output', str(output)])
    assert (output / 'audit.json').read_bytes() == saved


def test_cli_rejects_input_mutation_before_writing_evidence(tmp_path, monkeypatch):
    source = tmp_path / 'case.json'
    source.write_text(json.dumps({'record': record(NEGATIVE), 'branch': branch('e7d7', -3)}))
    original = audit.audit_mate_branch
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_text('{}')
        return result
    monkeypatch.setattr(audit, 'audit_mate_branch', changed)
    with pytest.raises(ValueError, match='changed during audit'):
        audit.main(['--input', str(source), '--output', str(tmp_path / 'output')])
    assert not (tmp_path / 'output').exists()
