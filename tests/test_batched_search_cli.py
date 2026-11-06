"""Native CLI orchestration with explicitly controlled CPU predictor and oracle."""
import json
import sys
import threading

import pytest

from xqgeneral.evidence import history_key, load_jsonl, write_jsonl
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, legal_moves, play, replay
from xqgeneral.sampling import REQUEST_SAMPLING_PROFILE
from test_stochastic_search_generation import POLICY


def fixture_inputs(tmp_path, monkeypatch):
    from xqgeneral import inference, search_distillation as module
    data = tmp_path / 'data'
    data.mkdir()
    rows = []
    line = ['b0c2', 'b9c7', 'a0a1']
    for i in range(4):
        history = replay(START_FEN, line[:i])
        rows.append({'id': str(i), 'game_id': 'controlled', 'split': 'train',
            'initial_fen': START_FEN, 'moves': line[:i], 'history': history, 'fen': history[-1],
            'feature_key': history_key(history), 'question': '受控测试', 'provenance': 'CPU_fixture_only'})
    write_jsonl(data / 'train.jsonl', rows)
    for split in ['validation', 'test']: write_jsonl(data / (split + '.jsonl'), [])
    paths = {}
    for name in ['checkpoint.pt', 'features.pt', 'engine', 'weights']:
        path = tmp_path / name
        path.write_bytes(b'controlled fixture; not real model or engine')
        paths[name] = path
    policy = tmp_path / 'generation.json'
    policy.write_text(json.dumps(POLICY))
    entered, closed, batches = [], [], []
    class Predictor:
        def generate(self, record, question, max_new_tokens, *, sampling, seed):
            assert max_new_tokens == POLICY['max_new_tokens']
            candidates = legal_moves(record['fen'])[:3]
            move = candidates[0]
            return json.dumps({'move': move, 'pv': [move], 'candidates': candidates,
                'facts': move_facts(record['fen'], move),
                'evaluation': {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'},
                'explanation': '这只是测试调度和原生账本的受控文本。'}, ensure_ascii=False)
        def generate_batch(self, records, *, max_new_tokens, sampling, seeds):
            batches.append(len(records))
            return [self.generate(r, r['question'], max_new_tokens, sampling=sampling, seed=s)
                    for r, s in zip(records, seeds)]
    class Oracle:
        def __init__(self, *args, **kwargs):
            self.owner = threading.get_ident()
            entered.append(self.owner)
        def analyze(self, fen, nodes, initial_fen, moves, searchmoves=None):
            assert threading.get_ident() == self.owner and replay(initial_fen, moves)[-1] == fen
            candidates = searchmoves or legal_moves(fen)[:3]
            return {'best_move': candidates[0], 'candidates': [
                {'move': move, 'pv': [move], 'score_type': 'cp', 'score': 0, 'wdl': [0, 1000, 0]}
                for move in candidates], 'requested_nodes': nodes, 'raw_output': ['controlled oracle']}
        def close(self):
            assert threading.get_ident() == self.owner
            closed.append(self.owner)
    monkeypatch.setattr(inference, 'Predictor', lambda *args, **kwargs: Predictor())
    monkeypatch.setattr(module, 'Pikafish', Oracle)
    args = ['search_distillation', '--checkpoint', str(paths['checkpoint.pt']), '--data', str(data),
        '--features', str(paths['features.pt']), '--executable', str(paths['engine']),
        '--weights', str(paths['weights']), '--student-generation-config', str(policy),
        '--limit', '4', '--nodes', '100', '--max-depth', '0', '--seed', '29']
    return module, args, entered, closed, batches


def test_real_cli_keeps_ordered_results_private_engines_and_raw_ledger(tmp_path, monkeypatch):
    module, args, entered, closed, batches = fixture_inputs(tmp_path, monkeypatch)
    output = tmp_path / 'batched'
    monkeypatch.setattr(sys, 'argv', args + ['--output', str(output),
        '--student-batch-size', '2', '--student-workers', '2', '--student-batch-wait-ms', '20'])
    module.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    proof = manifest['verification']
    assert proof['processed'] == 4 and proof['student_batching']['completed']
    assert proof['student_batching']['request_sampling_profile'] == REQUEST_SAMPLING_PROFILE
    assert proof['student_batching']['generated_requests'] == proof['stochastic_model_generations_this_session']
    assert len(entered) == len(closed) == 2 and sorted(entered) == sorted(closed)
    assert max(batches) == 2
    results = load_jsonl(output / 'results.partial.jsonl')
    assert len({r['source_root_id'] for r in results}) == len(results) == 4
    assert all(r['reason'] != 'mining_error' for r in results)
    oracle = load_jsonl(output / 'oracle-queries.partial.jsonl')
    assert len(oracle) == proof['oracle_query_records'] > 0
    assert all(replay(q['record']['initial_fen'], q['record']['moves'])[-1] == q['record']['fen'] for q in oracle)
    contract = json.loads((output / 'contract.json').read_text())
    assert contract['student_batching_profile']['profile'] == 'bounded_independent_root_workers_v1'
    decoded = load_jsonl(output / 'student-batches.partial.jsonl')
    assert len(decoded) == proof['student_decode_batch_records']
    assert [d['ordinal'] for d in decoded] == list(range(len(decoded)))
    assert sum(len(d['requests']) for d in decoded) == proof['student_batching']['generated_requests']
    assert proof['all_successful_student_decode_batches_preserved'] is True
    assert all(d['profile'] == REQUEST_SAMPLING_PROFILE and d['generation_policy'] == POLICY for d in decoded)
    assert all(isinstance(q['seed'], str) and replay(q['record']['initial_fen'], q['record']['moves'])[-1] == q['record']['fen']
               for d in decoded for q in d['requests'])


def test_default_cli_has_no_new_batch_arguments_or_profile(tmp_path, monkeypatch):
    module, args, entered, closed, batches = fixture_inputs(tmp_path, monkeypatch)
    output = tmp_path / 'serial'
    monkeypatch.setattr(sys, 'argv', args + ['--output', str(output)])
    module.main()
    contract = json.loads((output / 'contract.json').read_text())
    proof = json.loads((output / 'manifest.json').read_text())['verification']
    assert 'student_batching_profile' not in contract and 'student_batching' not in proof
    assert all(k not in contract['arguments'] for k in ['student_batch_size', 'student_workers', 'student_batch_wait_ms',
                                                       'student_autocast_cache'])
    assert 'student_autocast_cache_enabled' not in proof
    assert len(entered) == len(closed) == 1 and not batches
    assert proof['processed'] == 4


@pytest.mark.parametrize('setting,enabled', [('on', True), ('off', False)])
def test_native_mining_cli_binds_actual_explicit_student_cache_policy(tmp_path, monkeypatch, setting, enabled):
    from xqgeneral import inference
    module, args, entered, closed, batches = fixture_inputs(tmp_path, monkeypatch)
    original = inference.Predictor
    observed = []
    def predictor(*arguments, **kwargs):
        observed.append(kwargs['autocast_cache_enabled'])
        value = original(*arguments, **kwargs)
        value.config = {'autocast_cache_enabled': kwargs['autocast_cache_enabled']}
        return value
    monkeypatch.setattr(inference, 'Predictor', predictor)
    output = tmp_path/'explicit-cache'
    monkeypatch.setattr(sys, 'argv', args + ['--output', str(output), '--student-batch-size', '2',
                                         '--student-autocast-cache', setting])
    module.main()
    proof = json.loads((output/'manifest.json').read_text())
    assert observed == [enabled]
    assert proof['config']['student_autocast_cache'] == setting
    assert proof['verification']['student_autocast_cache_enabled'] is enabled
    assert proof['verification']['student_batching']['completed']


@pytest.mark.parametrize('extra', [['--student-batch-size', '0'], ['--student-workers', '2'],
    ['--student-batch-size', '4', '--student-workers', '2'], ['--student-batch-wait-ms', '10']])
def test_invalid_cli_batch_settings_fail_before_input_hash_or_model(tmp_path, monkeypatch, extra):
    from xqgeneral import search_distillation as module
    def forbidden(*args, **kwargs): raise AssertionError('Invalid options reached input hashing')
    monkeypatch.setattr(module, 'digest', forbidden)
    monkeypatch.setattr(sys, 'argv', ['search_distillation', '--checkpoint', 'absent.pt',
        '--output', str(tmp_path / 'absent'), *extra])
    with pytest.raises(ValueError): module.main()


@pytest.mark.parametrize('content', ['not valid JSON\n', json.dumps({
    'ordinal': 0, 'profile': 'wrong', 'generation_policy': POLICY, 'requests': []}) + '\n'])
def test_bad_resume_ledger_is_rejected_before_model_loading(tmp_path, monkeypatch, content):
    from xqgeneral import inference
    module, args, entered, closed, batches = fixture_inputs(tmp_path, monkeypatch)
    output = tmp_path / 'corrupt-ledger'
    output.mkdir()
    (output / 'student-batches.partial.jsonl').write_text(content)
    def forbidden(*args, **kwargs): raise AssertionError('Corrupt ledger reached model loading')
    monkeypatch.setattr(inference, 'Predictor', forbidden)
    monkeypatch.setattr(sys, 'argv', args + ['--output', str(output), '--student-batch-size', '2'])
    with pytest.raises(ValueError): module.main()
    assert not entered and not batches
