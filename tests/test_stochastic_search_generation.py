"""Sampling and resume controls; the decoder fixture is randomly initialized."""
import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xqgeneral.evidence import atomic_json, history_key
from xqgeneral.inference import Predictor
from xqgeneral.rules import START_FEN
from xqgeneral.sampling import generation_seed, sample_rng, sampling_options, search_generation_policy
from xqgeneral.search_distillation import SearchMiner


POLICY = {'do_sample': True, 'temperature': .6, 'top_k': 20, 'top_p': .95, 'max_new_tokens': 8}


@pytest.mark.parametrize('key,value', [('temperature', 0), ('temperature', float('nan')),
    ('temperature', True), ('top_k', True), ('top_k', -1), ('top_k', 1.5),
    ('top_p', 0), ('top_p', 1.1), ('top_p', float('inf')), ('do_sample', False),
    ('max_new_tokens', True), ('max_new_tokens', 0)])
def test_invalid_sampling_rejected(key, value):
    with pytest.raises(ValueError):
        search_generation_policy(dict(POLICY, **{key: value}))


def test_policy_is_explicit_and_does_not_modify_its_source():
    original = copy.deepcopy(POLICY)
    assert search_generation_policy(POLICY) == original and POLICY == original
    with pytest.raises(ValueError):
        search_generation_policy(dict(POLICY, unexpected=1))
    assert sampling_options(None) == {'do_sample': False}


@pytest.mark.parametrize('seed', [True, -1, 2**63, 1.5])
def test_invalid_seed_rejected_without_consuming_rng(seed):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError), sample_rng('cpu', seed):
        torch.rand(3)
    assert torch.equal(before, torch.get_rng_state())


def test_temporary_seed_restores_rng_on_success_and_exception():
    before = torch.get_rng_state().clone()
    with sample_rng('cpu', 91):
        first = torch.rand(16)
    with sample_rng('cpu', 91):
        assert torch.equal(torch.rand(16), first)
    with pytest.raises(RuntimeError), sample_rng('cpu', 22):
        torch.rand(5)
        raise RuntimeError('decoder failed')
    assert torch.equal(before, torch.get_rng_state())


@pytest.fixture
def neural_predictor():
    transformers = pytest.importorskip('transformers')
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(83).get_state())
        base = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=37,
            n_positions=64, n_embd=24, n_layer=1, n_head=2, resid_pdrop=0,
            embd_pdrop=0, attn_pdrop=0, bos_token_id=None, eos_token_id=None, pad_token_id=0)).eval()
    class Tokenizer:
        padding_side = 'right'
        pad_token_id = 0
        def apply_chat_template(self, *args, **kwargs): return 'controlled prompt'
        def __call__(self, texts, **kwargs):
            return transformers.BatchEncoding({'input_ids': torch.tensor([[1, 3, 5]] * len(texts)),
                'attention_mask': torch.ones((len(texts), 3), dtype=torch.long)})
        def decode(self, ids, **kwargs): return ' '.join(map(str, ids.tolist()))
    predictor = Predictor.__new__(Predictor)
    predictor.device, predictor.config, predictor.tokenizer = 'cpu', {'mode': 'bridge'}, Tokenizer()
    predictor.model = SimpleNamespace(base=base, board_context=lambda _: nullcontext())
    predictor.features = lambda _: []
    return predictor


def row(identity='one'):
    return {'id': identity, 'game_id': 'controlled', 'split': 'train', 'initial_fen': START_FEN,
        'fen': START_FEN, 'moves': [], 'history': [START_FEN], 'feature_key': history_key([START_FEN]),
        'question': '受控采样测试，不是棋力测量。'}


def reference_tokens(base, seed):
    """Independent full-prefix forward, nucleus selection and explicit generator."""
    ids = torch.tensor([[1, 3, 5]])
    generator = torch.Generator(device='cpu').manual_seed(seed)
    with torch.inference_mode():
        for _ in range(POLICY['max_new_tokens']):
            scores = base(ids).logits[:, -1].float() / POLICY['temperature']
            cutoff = torch.topk(scores, POLICY['top_k']).values[:, -1:]
            scores = scores.masked_fill(scores < cutoff, -torch.inf)
            values, order = scores.sort(descending=True)
            cumulative = values.softmax(-1).cumsum(-1)
            remove = cumulative > POLICY['top_p']
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            scores = scores.masked_fill(torch.zeros_like(remove).scatter(1, order, remove), -torch.inf)
            chosen = torch.multinomial(scores.softmax(-1), 1, generator=generator)
            ids = torch.cat([ids, chosen], -1)
    return ' '.join(map(str, ids[0, 3:].tolist()))


def test_actual_predictor_sampling_matches_independent_neural_decoding(neural_predictor):
    sampling = {k: v for k, v in POLICY.items() if k != 'max_new_tokens'}
    before = torch.get_rng_state().clone()
    answers = []
    for seed in [0, 19, 73]:
        actual = neural_predictor.generate(row(), max_new_tokens=8, sampling=sampling, seed=seed)
        assert actual == reference_tokens(neural_predictor.model.base, seed)
        answers.append(actual)
    assert len(set(answers)) > 1
    assert torch.equal(before, torch.get_rng_state())


def test_unspecified_generation_keeps_actual_greedy_prediction(neural_predictor):
    inputs = neural_predictor.tokenizer(['controlled prompt'])
    with torch.inference_mode():
        direct = neural_predictor.model.base.generate(**inputs, do_sample=False, max_new_tokens=8, pad_token_id=0)
    expected = neural_predictor.tokenizer.decode(direct[0, 3:])
    assert neural_predictor.generate(row(), max_new_tokens=8) == expected


def test_mining_is_identical_after_skipping_completed_roots_and_preserves_raw_errors(neural_predictor):
    class Oracle:
        def analyze(self, *args, **kwargs): raise AssertionError('Raw invalid generation must not be repaired')
    def run(records):
        miner = SearchMiner(neural_predictor, Oracle(), set(), generation_policy=POLICY, generation_seed=20260828)
        return {r['id']: miner.mine(r) for r in records}
    records = [row('one'), row('two'), row('three')]
    complete = run(records)
    resumed = {**run(records[:1]), **run(records[1:])}
    assert complete == resumed == run(list(reversed(records)))
    for query, trace, reason in complete.values():
        assert query is None and reason == 'invalid_root_candidates'
        assert trace[0]['root_generation']['raw']
        assert trace[0]['root_generation']['generation']['call_index'] == 0
    assert len({generation_seed(20260828, records[0], i) for i in range(20)}) == 20


def test_root_and_child_generations_record_distinct_ordinals_and_repeat_the_same_trace():
    from test_search_distillation import ControlledOracle, ControlledPredictor, root_record
    class PredictorWithSampling(ControlledPredictor):
        def generate(self, record, question, budget, *, sampling, seed):
            assert budget == POLICY['max_new_tokens'] and sampling['do_sample'] is True
            value = json.loads(super().generate(record, question, budget))
            suffix = int(torch.randint(100000, (1,), generator=torch.Generator().manual_seed(seed)))
            value['explanation'] += str(suffix)
            return json.dumps(value, ensure_ascii=False)
    miner = SearchMiner(PredictorWithSampling(), ControlledOracle(), set(),
        generation_policy=POLICY, generation_seed=20260828)
    first = miner.mine(root_record())
    assert first == miner.mine(root_record())
    query, trace, reason = first
    assert query is not None and reason == 'accepted_for_consolidation'
    generations = [trace[0]['root_generation']['generation'],
                   *[c['generation'] for c in trace[0]['children']]]
    # Children are ranked by quality after generation; ordinals retain call order.
    assert generations[0]['call_index'] == 0
    assert {c['move']: c['generation']['call_index'] for c in trace[0]['children']} == {
        'b0c2': 1, 'h0g2': 2, 'c3c4': 3}
    assert len({g['seed'] for g in generations}) == 4


def test_sampled_policy_is_bound_and_passed_through_existing_pilot_chain(tmp_path, monkeypatch):
    from test_search_pilot import ControlledChain, setup_pilot
    from xqgeneral import search_pilot
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    policy = tmp_path / 'sampling.json';atomic_json(policy, POLICY)
    settings = json.loads(config.read_text());settings['student_generation_config'] = str(policy)
    atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, query)
    output = tmp_path / 'pilot'
    search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    command = next(c for c in chain.commands if 'xqgeneral.search_distillation' in c)
    assert command[command.index('--student-generation-config') + 1] == str(policy)
    proof = json.loads((output / 'mining/manifest.json').read_text())
    assert proof['config']['student_generation_config'] == str(policy)
    assert str(policy) in proof['inputs']
    atomic_json(policy, dict(POLICY, temperature=.7))
    with pytest.raises(ValueError):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output, resume=True)
    assert chain.student_loads == 1


def test_mining_resume_binds_every_expert_feature_shard_before_student_load(tmp_path, monkeypatch):
    from test_feature_store import prepared
    from test_search_pilot import ControlledChain, setup_pilot, artifact
    from xqgeneral import search_distillation, search_pilot
    from xqgeneral.feature_store import feature_paths
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    _, store, _, _ = prepared(tmp_path)
    settings = json.loads(config.read_text())
    evaluation_path = Path(settings['evaluation_config']);evaluation = json.loads(evaluation_path.read_text())
    recipe_path = Path(evaluation['sft_recipe']);recipe = json.loads(recipe_path.read_text())
    recipe['feature_path'] = evaluation['features'] = str(store)
    atomic_json(recipe_path, recipe);atomic_json(evaluation_path, evaluation)
    token_path = Path(evaluation['token_preflight']);token = json.loads(token_path.read_text())
    token['cache_header_only_input'] = {'path': str(store), **artifact(store)};atomic_json(token_path, token)
    capability = json.loads((validation / 'manifest.json').read_text())
    capability['config']['config_sha256'] = artifact(evaluation_path)['sha256']
    atomic_json(validation / 'manifest.json', capability)
    chain = ControlledChain(monkeypatch, query)
    output = tmp_path / 'pilot'
    search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    mining = output / 'mining';proof = json.loads((mining / 'manifest.json').read_text())
    assert all(str(p) in proof['inputs'] for p in feature_paths(store))
    (mining / 'manifest.json').unlink()  # Retain the completed root journal as an interrupted-finalization fixture.
    shard = Path(feature_paths(store)[-1]);raw = shard.read_bytes();shard.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    command = next(c for c in chain.commands if 'xqgeneral.search_distillation' in c)
    module_index = command.index('xqgeneral.search_distillation')
    monkeypatch.setattr(sys, 'argv', command[module_index:])
    with pytest.raises(ValueError, match='continuation inputs'):
        search_distillation.main()
    assert chain.student_loads == 1
