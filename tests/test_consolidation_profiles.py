"""Teacher input contracts; controlled CPU models do not establish semantic quality."""
import copy
import hashlib
import json

import pytest

from xqgeneral import prose_moves, search_pilot
from xqgeneral.evidence import atomic_json
from xqgeneral.rules import START_FEN
from test_search_child_prompts import CapturingPredictor, make_search
from test_search_distillation import root_record
from test_search_pilot import ControlledChain, setup_pilot


RAW = 'raw_child_prose_v1'
VERIFIED = 'verified_evidence_v1'


def packet_and_target():
    packet = {'child_analyses': [{'move': 'b0c2', 'terminal_outcome': None,
        'recommended_move': 'b9c7', 'unverified_explanation': '原始子讲解：黑马跳出，仍需核对。'}]}
    target = {'pv': ['b0c2', 'b9c7'], 'branches': [{'pv': ['b0c2', 'b9c7']}]}
    return packet, target


def test_implicit_profile_preserves_original_message_identity():
    packet, target = packet_and_target()
    message = prose_moves.consolidation_messages(packet, START_FEN, target)
    # Captured from the unchanged completed-pilot implementation before this feature.
    assert hashlib.sha256(json.dumps(message, ensure_ascii=False).encode()).hexdigest() == '023b6feba7c66ffd0002c6c425104e443aa6527e54449c934819abb68bbf0c15'
    payload = json.loads(message[-1]['content'])
    assert 'consolidation_profile' not in payload


@pytest.mark.parametrize('schema', ['move_eval', 'full'])
def test_explicit_raw_profile_preserves_actual_child_string_and_input(schema):
    packet, target = packet_and_target()
    original_text = packet['child_analyses'][0]['unverified_explanation']
    if schema == 'full':
        packet = {'verified_child_analyses': [{'move': 'b0c2', 'terminal_outcome': None,
            'analysis': {'move': 'b9c7', 'explanation': original_text}}]}
    before = copy.deepcopy(packet)
    messages = prose_moves.consolidation_messages(packet, START_FEN, target, consolidation_profile=RAW)
    payload = json.loads(messages[-1]['content'])
    assert packet == before
    if schema == 'move_eval':
        assert payload['child_analyses'][0]['unverified_explanation'] == original_text
    else:
        assert payload['verified_child_analyses'][0]['analysis']['explanation'] == original_text
    assert payload['consolidation_profile'] == RAW
    assert payload['unverified_child_prose_omitted'] is False
    assert payload['child_prose_evidence']['children_with_prose'] == 1
    assert '已省略' not in messages[0]['content']
    assert '未经语义核验' in messages[0]['content']


def test_explicit_verified_profile_removes_text_and_attests_to_actual_packet():
    packet, target = packet_and_target()
    payload = json.loads(prose_moves.consolidation_messages(
        packet, START_FEN, target, consolidation_profile=VERIFIED)[-1]['content'])
    assert 'unverified_explanation' not in payload['child_analyses'][0]
    assert payload['consolidation_profile'] == VERIFIED
    assert payload['unverified_child_prose_omitted'] is True
    assert payload['child_prose_evidence']['children_with_prose'] == 0


@pytest.mark.parametrize('text', [None, '', '   ', 7])
def test_raw_profile_requires_prose_for_each_nonterminal_child(text):
    packet, target = packet_and_target()
    packet['child_analyses'][0]['unverified_explanation'] = text
    with pytest.raises(ValueError, match='child prose'):
        prose_moves.consolidation_messages(packet, START_FEN, target, consolidation_profile=RAW)


def test_terminal_child_needs_no_fabricated_model_explanation():
    packet, target = packet_and_target()
    child = packet['child_analyses'][0]
    child.update(terminal_outcome={'ended': True, 'winner': 'red', 'reason': 'checkmate'},
                 recommended_move=None, unverified_explanation=None)
    payload = json.loads(prose_moves.consolidation_messages(
        packet, START_FEN, target, consolidation_profile=RAW)[-1]['content'])
    assert payload['child_analyses'][0]['unverified_explanation'] is None
    assert payload['child_prose_evidence']['terminal_children_without_prose'] == 1


@pytest.mark.parametrize('profile', ['', 'raw_child_prose_v0', True, 1, {}])
def test_unknown_profile_cannot_silently_change_teacher_input(profile):
    packet, target = packet_and_target()
    with pytest.raises(ValueError, match='consolidation profile'):
        prose_moves.consolidation_messages(packet, START_FEN, target, consolidation_profile=profile)


def test_raw_profile_rejects_numerical_only_child_prompt_before_student_queries():
    predictor = CapturingPredictor()
    with pytest.raises(ValueError, match='full_explanation'):
        make_search(predictor, 'paper_recursive', consolidation_profile=RAW,
                    child_prompt_profile='numeric_six_ply_v1')
    assert not predictor.calls


def test_miner_transmits_raw_text_without_changing_search_or_rule_targets():
    original = make_search(CapturingPredictor(), 'paper_recursive').mine(root_record())
    actual = make_search(CapturingPredictor(), 'paper_recursive', consolidation_profile=RAW).mine(root_record())
    query, trace, reason = actual
    assert reason == original[2] == 'accepted_for_consolidation'
    assert trace == original[1]
    assert query['target_fields'] == original[0]['target_fields']
    payload = json.loads(query['messages'][-1]['content'])
    expected = {c['move']: c['analysis']['explanation'] for c in trace[-1]['children'] if c.get('analysis')}
    assert expected
    assert {c['move']: c['unverified_explanation'] for c in payload['child_analyses']
            if c.get('unverified_explanation')} == expected


@pytest.mark.parametrize('acceptance', ['strict', 'paper_recursive'])
def test_miner_preserves_failed_raw_trace_when_a_child_omits_requested_prose(acceptance):
    class MissingProse(CapturingPredictor):
        def generate(self, *args, **kwargs):
            raw = json.loads(super().generate(*args, **kwargs))
            raw.pop('explanation', None)
            return json.dumps(raw, ensure_ascii=False)
    miner = make_search(MissingProse(), acceptance, consolidation_profile=RAW)
    query, trace, reason = miner.mine(root_record())
    assert query is None and reason == 'missing_child_prose'
    assert trace and trace[-1]['children']
    assert all('explanation' not in json.loads(c['raw']) for c in trace[-1]['children'])
    assert miner.counts['missing_child_prose_roots'] == 1


@pytest.mark.parametrize('profile', [None, '', False, 1, 'bogus'])
def test_bad_pilot_profile_is_rejected_before_models_or_teacher(tmp_path, monkeypatch, profile):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    settings['consolidation_profile'] = profile
    atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, query)
    with pytest.raises(ValueError, match='consolidation profile'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert chain.student_loads == chain.teacher_loads == 0 and not chain.commands


def query_with_profile(query, profile):
    query = copy.deepcopy(query)
    packet = {'root_fen': query['record']['fen'], 'target_fields': query['target_fields'],
        'child_analyses': [{'move': query['target_fields']['move'], 'terminal_outcome': None,
                            'unverified_explanation': '受控CPU原文，不作真实讲解质量成绩。'}]}
    for child in packet.get('child_analyses', []):
        child['unverified_explanation'] = '受控CPU子讲解原文，规则事实仍须独立核对。'
    for child in packet.get('verified_child_analyses', []):
        child['analysis'] = dict(child.get('analysis') or {}, explanation='受控CPU子讲解原文。')
    query['messages'] = prose_moves.consolidation_messages(
        packet, query['record']['fen'], query['target_fields'], consolidation_profile=profile)
    return query


@pytest.mark.parametrize('profile', [RAW, VERIFIED, 'grounded_child_prose_v2',
                                     'grounded_terminal_child_prose_v3'])
def test_profile_cache_batch_and_strict_collection_survive_native_CLI_chain(tmp_path, monkeypatch, profile):
    from test_stochastic_search_generation import POLICY
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    policy = tmp_path / 'student-generation.json'
    atomic_json(policy, POLICY)
    settings.update(consolidation_profile=profile, student_batch_size=2, student_workers=2,
        student_generation_config=str(policy), student_autocast_cache_enabled=True,
        require_immediate_mate_consistency=True)
    atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, query_with_profile(query, profile))
    out = tmp_path / 'pilot'
    result = search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, out)
    assert result['accepted_structured_training_labels'] == 1
    command = chain.commands[0]
    assert command[command.index('--consolidation-profile') + 1] == profile
    proof = json.loads((out / 'mining/manifest.json').read_text())
    assert proof['verification']['consolidation_profile'] == profile
    evidence = proof['verification']['consolidation_child_prose_evidence']
    assert evidence['profile'] == profile and evidence['queries'] == 1
    assert bool(evidence['children_with_prose']) == (profile != VERIFIED)
    assert result == search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, out, resume=True)
    assert len(chain.commands) == 3 and chain.student_loads == chain.teacher_loads == 1


@pytest.mark.parametrize('problem', ['profile', 'counts'])
def test_mining_profile_proof_mismatch_blocks_teacher(tmp_path, monkeypatch, problem):
    from pathlib import Path
    class MismatchedChain(ControlledChain):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if 'xqgeneral.search_distillation' in command:
                out = Path(command[command.index('--output') + 1])
                proof = json.loads((out / 'manifest.json').read_text())
                if problem == 'profile':
                    proof['verification']['consolidation_profile'] = VERIFIED
                else:
                    proof['verification']['consolidation_child_prose_evidence']['children_with_prose'] += 1
                atomic_json(out / 'manifest.json', proof)
            return result
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    settings['consolidation_profile'] = RAW
    atomic_json(config, settings)
    chain = MismatchedChain(monkeypatch, query_with_profile(query, RAW))
    with pytest.raises(ValueError, match='consolidation profile'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert len(chain.commands) == 1 and chain.student_loads == 1 and chain.teacher_loads == 0


def test_default_profile_does_not_inherit_a_caller_supplied_raw_profile_tag():
    packet, target = packet_and_target()
    packet['consolidation_profile'] = RAW
    packet['child_prose_evidence'] = {'children_with_prose': 99}
    actual = json.loads(prose_moves.consolidation_messages(packet, START_FEN, target)[-1]['content'])
    assert 'consolidation_profile' not in actual and 'child_prose_evidence' not in actual
    assert actual['unverified_child_prose_omitted'] is True
