"""Controlled CLI chain tests; no trained student, real teacher or engine claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_search_collection import recorded_collection, response, TEACHER
from xqgeneral import collect_search, local_teacher, search_distillation, search_pilot
from xqgeneral.evidence import atomic_json, code_identity, digest, load_jsonl, manifest


def artifact(path):
    return {'sha256': digest(path), 'bytes': Path(path).stat().st_size}


def setup_pilot(tmp_path, monkeypatch, complete=True):
    args, pool, _, query = recorded_collection(tmp_path)
    cache = tmp_path / 'controlled-cache.pt'
    cache.write_bytes(b'controlled identity only; not an expert feature cache')
    recipe = tmp_path / 'recipe.json'
    atomic_json(recipe, {'require_clean_foundation_handoff': True, 'data_path': str(args['heldout_data']),
                         'feature_path': str(cache)})
    token = tmp_path / 'token.json'
    atomic_json(token, {'status': 'complete', 'kind': 'latent_only_reviewed_explanation_data_preflight',
        'config': {'data': str(args['heldout_data'])}, 'inputs': {}, 'outputs': {},
        'cache_header_only_input': {'path': str(cache), **artifact(cache)}})
    teacher_root = tmp_path / 'controlled-teacher'
    teacher_root.mkdir()
    shard = teacher_root / 'model.safetensors'
    shard.write_bytes(b'controlled BF16 identity; not real model weights')
    teacher = dict(TEACHER, model_path=str(teacher_root))
    atomic_json(teacher_root / 'weights.manifest.json', dict(teacher,
        all_official_lfs_sha256_matched=True, stored_tensor_dtypes={'BF16': 1},
        shards={'model.safetensors': artifact(shard)}, total_weight_bytes=shard.stat().st_size))
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        (teacher_root / name).write_text('{}')
    teacher_config = tmp_path / 'teachers.json'
    atomic_json(teacher_config, {'search_consolidator': teacher})
    engine, weights = tmp_path / 'controlled-engine', tmp_path / 'controlled.nnue'
    engine.write_bytes(b'controlled engine identity')
    weights.write_bytes(b'controlled engine weights identity')
    evaluation = tmp_path / 'evaluation.json'
    atomic_json(evaluation, {'sft_recipe': str(recipe), 'token_preflight': str(token),
        'data': str(args['heldout_data']), 'features': str(cache), 'split': 'validation',
        'memories': ['normal', 'zero', 'shuffled'], 'validation_examples': 1,
        'teacher_config': str(teacher_config), 'executable': str(engine), 'weights': str(weights)})
    config = tmp_path / 'pilot.json'
    atomic_json(config, {'evaluation_config': str(evaluation), 'data': str(args['data']),
        'recorded_inputs': str(pool), 'limit': 1, 'nodes': 100000, 'max_depth': 5,
        'child_contract': 'move_eval', 'seed': 20261013, 'teacher_max_new_tokens': 1536,
        'cuda_visible_devices': '0'})
    pipeline = tmp_path / 'pipeline'
    checkpoint = pipeline / 'sft/adapter.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'controlled parent identity; not a trained model')
    parent = manifest('controlled_parent_for_boundary_test', {}, [], [checkpoint], {'selected_sft_step': 9})
    atomic_json(pipeline / 'manifest.json', parent)
    validation = tmp_path / 'validation'
    validation.mkdir()
    verification = {'actual_clean_initial_sft_completed_before_model_or_reviewer_load': True,
        'split': 'validation', 'all_validation_examples': 1,
        'independent_test_used_for_training_or_selection': False, 'raw_answers_repaired': False,
        'raw_validation_by_memory': {name: {} for name in ['normal', 'zero', 'shuffled']}}
    capability = manifest('clean_initial_sft_capability_validation',
        {'pipeline': str(pipeline), 'config': str(evaluation), 'config_sha256': digest(evaluation)},
        [checkpoint], [], verification)
    if complete:
        atomic_json(validation / 'manifest.json', capability)
    # The real checked_sft seam is covered by test_clean_sft_evaluation. This
    # boundary fixture keeps orchestration tests free of real decoder states.
    monkeypatch.setattr(search_pilot, 'checked_sft', lambda *a: (checkpoint, parent))
    query['messages'] = [{'role': 'user', 'content': '受控汇总查询，非真实学生搜索。'}]
    return pipeline, validation, config, query, capability


class ControlledChain:
    def __init__(self, monkeypatch, query, *, empty=False):
        self.monkeypatch, self.query, self.empty = monkeypatch, query, empty
        self.commands, self.student_loads, self.teacher_loads = [], 0, 0
        self.failure = None
        self.original = subprocess.run
        owner = self
        class Predictor:
            def __init__(self, *args, **kwargs):
                owner.student_loads += 1
        class Oracle:
            def __init__(self, *args, **kwargs):
                pass
            def close(self):
                pass
        class Teacher:
            def __init__(self, config):
                owner.teacher_loads += 1
                self.identity = response()['teacher']
            def generate(self, *args):
                return {key: response()[key] for key in ['text', 'hit_generation_limit']}
        from xqgeneral import inference
        monkeypatch.setattr(inference, 'Predictor', Predictor)
        monkeypatch.setattr(search_distillation, 'Pikafish', Oracle)
        monkeypatch.setattr(local_teacher, 'LocalConsolidator', Teacher)
        def mine(miner, row):
            assert row['id'] == owner.query['source_root_id']
            return (None, [], 'invalid_root_candidates') if owner.empty else (
                copy.deepcopy(owner.query), [], 'accepted_for_consolidation')
        monkeypatch.setattr(search_distillation.SearchMiner, 'mine', mine)
        monkeypatch.setattr(subprocess, 'run', self)

    def __call__(self, command, **kwargs):
        if command[0] in ['git', 'tmux']:
            return self.original(command, **kwargs)
        self.commands.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'
        module = command[command.index('-m') + 1]
        argv = command[command.index(module) + 1:]
        implementation = {'xqgeneral.search_distillation': search_distillation,
                          'xqgeneral.local_teacher': local_teacher,
                          'xqgeneral.collect_search': collect_search}[module]
        with self.monkeypatch.context() as patch:
            patch.setattr(sys, 'argv', [module, *argv])
            implementation.main()
        if module == 'xqgeneral.search_distillation' and self.failure:
            root = Path(argv[argv.index('--output') + 1])
            path = root / 'manifest.json'
            proof = json.loads(path.read_text())
            if self.failure == 'false_yield':
                proof['verification']['accepted_for_consolidation'] += 1
            elif self.failure == 'lost_isolation':
                proof['verification']['raw_child_reachable_prefixes_isolated'] = False
            elif self.failure == 'wrong_source':
                proof['code'] = {'revision': 'unrelated'}
            elif self.failure == 'changed_queries':
                with (root / 'queries.jsonl').open('a') as handle:
                    handle.write('{}\n')
            atomic_json(path, proof)
        return SimpleNamespace(returncode=0)


def test_actual_cli_chain_counts_labels_and_preserves_original_heldouts(tmp_path, monkeypatch):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    chain = ControlledChain(monkeypatch, query)
    output = tmp_path / 'pilot'
    result = search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    assert len(chain.commands) == 3 and chain.student_loads == chain.teacher_loads == 1
    assert result['attempted_original_training_roots'] == result['mined_queries'] == 1
    assert result['accepted_structured_training_labels'] == 1 and result['final_structured_label_yield'] == 1
    assert not result['strategic_prose_fully_verified'] and result['trained_search_distillation_rounds'] == 0
    evaluation = json.loads(Path(json.loads(config.read_text())['evaluation_config']).read_text())
    for split in ['validation', 'test']:
        assert (output / 'labels' / (split + '.jsonl')).read_bytes() == (
            Path(evaluation['data']) / (split + '.jsonl')).read_bytes()
    assert result == search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output, resume=True)
    assert len(chain.commands) == 3
    (output / 'labels/train.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output, resume=True)


def test_no_improvements_skip_teacher_collection_and_training(tmp_path, monkeypatch):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    chain = ControlledChain(monkeypatch, query, empty=True)
    output = tmp_path / 'pilot'
    result = search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    assert chain.student_loads == 1 and chain.teacher_loads == 0 and len(chain.commands) == 1
    assert result['mining_reasons'] == {'invalid_root_candidates': 1}
    assert result['final_structured_label_yield'] == 0 and not result['neural_consolidation_executed']
    assert result['trained_search_distillation_rounds'] == 0 and not (output / 'labels').exists()


def test_stopped_owned_validation_refuses_search_before_any_model_load(tmp_path, monkeypatch):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch, complete=False)
    chain = ControlledChain(monkeypatch, query)
    def stopped(command, **kwargs):
        if command[0] == 'tmux':
            assert command == ['tmux', 'has-session', '-t', '=owned-validation']
            return SimpleNamespace(returncode=1)
        return chain(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', stopped)
    with pytest.raises(RuntimeError, match='stopped before'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert chain.student_loads == chain.teacher_loads == 0 and not chain.commands


def test_live_wait_transitions_only_after_exact_capability_completion(tmp_path, monkeypatch):
    pipeline, validation, config, query, capability = setup_pilot(tmp_path, monkeypatch, complete=False)
    chain = ControlledChain(monkeypatch, query)
    sleeps = []
    def alive(command, **kwargs):
        if command[0] == 'tmux':
            return SimpleNamespace(returncode=0)
        return chain(command, **kwargs)
    def complete(seconds):
        assert not chain.commands and chain.student_loads == chain.teacher_loads == 0
        state = json.loads((tmp_path / 'pilot/state.json').read_text())
        assert state['student_or_teacher_loaded'] is False
        sleeps.append(seconds)
        atomic_json(validation / 'manifest.json', capability)
    monkeypatch.setattr(subprocess, 'run', alive)
    monkeypatch.setattr(search_pilot.time, 'sleep', complete)
    result = search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot', poll_seconds=1)
    assert sleeps == [1] and len(chain.commands) == 3 and result['accepted_structured_training_labels'] == 1


def test_changed_config_during_live_wait_refused_before_models(tmp_path, monkeypatch):
    pipeline, validation, config, query, capability = setup_pilot(tmp_path, monkeypatch, complete=False)
    chain = ControlledChain(monkeypatch, query)
    def alive(command, **kwargs):
        return SimpleNamespace(returncode=0) if command[0] == 'tmux' else chain(command, **kwargs)
    def changed(seconds):
        value = json.loads(config.read_text())
        value['nodes'] += 1
        atomic_json(config, value)
        atomic_json(validation / 'manifest.json', capability)
    monkeypatch.setattr(subprocess, 'run', alive)
    monkeypatch.setattr(search_pilot.time, 'sleep', changed)
    with pytest.raises(ValueError, match='changed'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot', poll_seconds=1)
    assert not chain.commands and chain.student_loads == chain.teacher_loads == 0


@pytest.mark.parametrize('key,value', [('limit', 0), ('nodes', True), ('max_depth', -1),
                                     ('teacher_max_new_tokens', 0), ('child_contract', 'unverified')])
def test_invalid_budgets_refused_before_wait_or_model_load(tmp_path, monkeypatch, key, value):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    saved = json.loads(config.read_text())
    saved[key] = value
    atomic_json(config, saved)
    chain = ControlledChain(monkeypatch, query)
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    assert not output.exists() and not chain.commands and chain.student_loads == chain.teacher_loads == 0


@pytest.mark.parametrize('problem', ['other_parent', 'missing_memory', 'test_selection', 'configuration_changed'])
def test_incomplete_or_changed_capability_refused_before_search(tmp_path, monkeypatch, problem):
    pipeline, validation, config, query, capability = setup_pilot(tmp_path, monkeypatch)
    if problem == 'other_parent':
        capability['inputs'] = {}
    elif problem == 'missing_memory':
        capability['verification']['raw_validation_by_memory'].pop('zero')
    elif problem == 'test_selection':
        capability['verification']['independent_test_used_for_training_or_selection'] = True
    else:
        capability['config']['config_sha256'] = '0' * 64
    atomic_json(validation / 'manifest.json', capability)
    chain = ControlledChain(monkeypatch, query)
    with pytest.raises(ValueError):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert not chain.commands and chain.student_loads == chain.teacher_loads == 0


@pytest.mark.parametrize('failure', ['false_yield', 'lost_isolation', 'wrong_source', 'changed_queries'])
def test_bad_search_output_never_reaches_teacher(tmp_path, monkeypatch, failure):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    chain = ControlledChain(monkeypatch, query)
    chain.failure = failure
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    assert len(chain.commands) == 1 and chain.teacher_loads == 0
    assert json.loads((output / 'state.json').read_text())['stage'] == 'student_recursive_search'
    assert (output / 'mining/manifest.json').exists() and not (output / 'manifest.json').exists()
