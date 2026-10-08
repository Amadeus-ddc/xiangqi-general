"""Controlled CPU execution of the clean SFT validation and neural-review chain."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_sft_pipeline import install_runner, setup_data
from xqgeneral import clean_sft_evaluation as evaluation
from xqgeneral import evaluate_explanations, inference, review_explanations, sft_pipeline
from xqgeneral.evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from xqgeneral.rules import START_FEN


def artifact(path):
    return {'sha256': digest(path), 'bytes': Path(path).stat().st_size}


def setup_evaluation(tmp_path, monkeypatch, complete=True):
    curriculum, recipe_path, token_path = setup_data(tmp_path)
    recipe = json.loads(recipe_path.read_text());token = json.loads(token_path.read_text())
    for split in ['train', 'validation']:
        path = Path(recipe['data_path']) / (split + '.jsonl');rows = load_jsonl(path)
        for row in rows:
            row.update(feature_key=row['id'], fen=START_FEN, initial_fen=START_FEN, moves=[], question='preserved question')
        write_jsonl(path, rows);token['inputs'][str(path)] = artifact(path)
    atomic_json(token_path, token)
    training_runner = install_runner(monkeypatch)
    pipeline = tmp_path / 'initial-sft'
    if complete:sft_pipeline.run_pipeline(curriculum, 'owned-curriculum', recipe_path, token_path, pipeline)
    teacher_root = tmp_path / 'teacher';teacher_root.mkdir()
    shard = teacher_root / 'model.safetensors';shard.write_bytes(b'controlled BF16 weight identity; not real model weights')
    teacher = {'repository': 'Qwen/Qwen3.8-27B', 'revision': 'controlled-fixture-only', 'quantization': 'none',
               'inference_dtype': 'bfloat16', 'model_path': str(teacher_root)}
    atomic_json(teacher_root / 'weights.manifest.json', {'repository': teacher['repository'],
        'revision': teacher['revision'], 'quantization': 'none', 'all_official_lfs_sha256_matched': True,
        'stored_tensor_dtypes': {'BF16': 1}, 'total_weight_bytes': shard.stat().st_size,
        'shards': {'model.safetensors': artifact(shard)}})
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:(teacher_root / name).write_text('{}')
    teacher_path = tmp_path / 'teacher.json';atomic_json(teacher_path, {'search_consolidator': teacher})
    engine = tmp_path / 'controlled-engine';engine.write_bytes(b'controlled engine identity')
    weights = tmp_path / 'controlled.nnue';weights.write_bytes(b'controlled engine weight identity')
    config = json.loads(Path('configs/evaluation-clean-sft-v1.json').read_text())
    config.update(sft_recipe=str(recipe_path), token_preflight=str(token_path), data=recipe['data_path'],
                  features=recipe['feature_path'], validation_examples=16, teacher_config=str(teacher_path),
                  executable=str(engine), weights=str(weights), workers=1)
    path = tmp_path / 'evaluation.json';atomic_json(path, config)
    return pipeline, path, training_runner


class ControlledEvaluation:
    def __init__(self, original, monkeypatch):
        self.original = original;self.patch = monkeypatch;self.commands = [];self.batches = [];self.loads = 0;self.failure = None
        owner = self
        class Predictor:
            def __init__(self, *args, **kwargs):owner.loads += 1
            def generate_batch(self, rows, max_new_tokens, memory='normal'):
                owner.batches.append((memory, [row['id'] for row in rows], max_new_tokens))
                return ['unmodified invalid student JSON'] * (len(rows) - int(owner.failure == 'short_batch'))
        monkeypatch.setattr(inference, 'Predictor', Predictor)
        monkeypatch.setattr(evaluate_explanations, 'judge_shard', lambda rows, args:
            [dict(row, judgment=evaluate_explanations.judgment(None, row['record'], row['raw'], args.nodes)) for row in rows])
        monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
        monkeypatch.setattr(subprocess, 'run', self)

    def __call__(self, command, **kwargs):
        if command[0] in ['git', 'tmux']:return self.original(command, **kwargs)
        self.commands.append(command)
        module = command[command.index('-m') + 1];arguments = command[command.index(module) + 1:]
        if module == 'xqgeneral.evaluate_explanations':
            args = {arguments[i][2:].replace('-', '_'): arguments[i + 1] for i in range(0, len(arguments), 2)}
            for key in ['limit', 'batch_size', 'max_new_tokens', 'nodes', 'workers', 'seed']:args[key] = int(args[key])
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'
            evaluate_explanations.run_evaluation(SimpleNamespace(**args));root = Path(args['output'])
            proof = json.loads((root / 'manifest.json').read_text())
            if self.failure == 'changed_judged_raw':
                path = root / 'judged-predictions.jsonl';rows = load_jsonl(path);rows[0]['raw'] = 'repaired answer';write_jsonl(path, rows)
                proof['outputs'][str(path)] = artifact(path)
            elif self.failure == 'false_metrics':
                path = root / 'metrics.json';metrics = json.loads(path.read_text());metrics['first_move_no_mistake_rate'] = 1
                atomic_json(path, metrics);proof['verification'] = metrics;proof['outputs'][str(path)] = artifact(path)
            elif self.failure == 'wrong_memory':proof['config']['memory'] = 'zero'
            atomic_json(root / 'manifest.json', proof)
        elif module == 'xqgeneral.review_explanations':
            with self.patch.context() as patch:
                patch.setattr(sys, 'argv', ['review', *arguments]);review_explanations.main()
        elif module == 'xqgeneral.local_teacher':
            args = {arguments[i][2:].replace('-', '_'): arguments[i + 1] for i in range(0, len(arguments), 2)}
            args['max_new_tokens'] = int(args['max_new_tokens']);root = Path(args['output'])
            teacher = json.loads(Path(args['config']).read_text())['search_consolidator']
            queries = load_jsonl(args['input'])
            identity = dict(teacher, parameter_elements_by_dtype={'torch.bfloat16': 27})
            rating = {'factual_correctness': 2, 'strategic_reasoning': 2, 'clarity': 3, 'instruction_adherence': 1,
                      'unsupported_claims': ['原始回答缺少可核验的推荐走法。'], 'reason': '原始回答不是有效讲解，无法支持其走法与战略结论。'}
            responses = [{'id': q['id'], 'teacher': identity, 'text': json.dumps(rating, ensure_ascii=False),
                          'hit_generation_limit': i == 0} for i, q in enumerate(queries)]
            if self.failure == 'foreign_response':responses[0]['id'] = 'foreign'
            write_jsonl(root / 'responses.jsonl', responses);atomic_json(root / 'deployment.json', identity)
            saved_args = dict(args)
            if self.failure == 'foreign_reviewer_input':saved_args['input'] = 'unrelated-queries.jsonl'
            atomic_json(root / 'manifest.json', manifest('local_teacher_inference', saved_args,
                [args['input'], args['config'], Path(teacher['model_path']) / 'weights.manifest.json'],
                [root / 'responses.jsonl', root / 'deployment.json'], {'examples': len(responses), 'teacher': identity}))
        else:raise AssertionError('Unexpected controlled command: ' + module)
        return SimpleNamespace(returncode=0)


def test_complete_paired_raw_validation_and_full_blinded_review_preserve_errors(tmp_path, monkeypatch):
    pipeline, config, original = setup_evaluation(tmp_path, monkeypatch)
    controlled = ControlledEvaluation(original, monkeypatch);output = tmp_path / 'validation'
    result = evaluation.run_evaluation(pipeline, 'owned-sft', config, output)
    assert len(controlled.commands) == 6 and controlled.loads == 3
    assert {memory for memory, _, _ in controlled.batches} == {'normal', 'zero', 'shuffled'}
    ids = [[ids for memory, ids, _ in controlled.batches if memory == value] for value in ['normal', 'zero', 'shuffled']]
    assert ids[0] == ids[1] == ids[2]
    assert result['all_validation_examples'] == 16 and result['split'] == 'validation'
    assert all(m['examples'] == 16 and m['parse_valid_rate'] == 0 and m['oracle_repairs'] == 0
               for m in result['raw_validation_by_memory'].values())
    assert result['blinded_prose_review']['valid_ratings'] == 15 and result['blinded_prose_review']['rejected'] == 1
    assert result['blinded_prose_review']['all_dimensions_at_least_4_fraction_all_queries'] == 0
    assert not result['strong_play_or_reliable_coaching_proven'] and not result['raw_answers_repaired']
    assert result == evaluation.run_evaluation(pipeline, 'owned-sft', config, output, resume=True)
    assert len(controlled.commands) == 6
    path = output / 'normal/raw-predictions.jsonl';path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        evaluation.run_evaluation(pipeline, 'owned-sft', config, output, resume=True)


def test_missing_owned_sft_pipeline_never_loads_model_or_reviewer(tmp_path, monkeypatch):
    pipeline, config, original = setup_evaluation(tmp_path, monkeypatch, complete=False)
    controlled = ControlledEvaluation(original, monkeypatch)
    def missing(command, **kwargs):
        if command[0] == 'tmux':
            assert command == ['tmux', 'has-session', '-t', '=owned-sft'];return SimpleNamespace(returncode=1)
        return controlled(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', missing)
    output = tmp_path / 'waiting'
    with pytest.raises(RuntimeError, match='stopped before'):
        evaluation.run_evaluation(pipeline, 'owned-sft', config, output)
    assert not controlled.commands and controlled.loads == 0 and not (output / 'normal').exists()
    assert json.loads((output / 'state.json').read_text())['stage'] == 'waiting_for_completed_clean_sft'


def test_changed_configuration_while_waiting_refused_before_model_load(tmp_path, monkeypatch):
    pipeline, config, original = setup_evaluation(tmp_path, monkeypatch)
    controlled = ControlledEvaluation(original, monkeypatch);path = pipeline / 'manifest.json';preserved = path.read_bytes();path.unlink()
    def advance(seconds):
        value = json.loads(config.read_text());value['nodes'] += 1;atomic_json(config, value);path.write_bytes(preserved)
    monkeypatch.setattr(evaluation.time, 'sleep', advance)
    with pytest.raises(ValueError, match='configuration'):
        evaluation.run_evaluation(pipeline, 'owned-sft', config, tmp_path / 'changed', poll_seconds=1)
    assert not controlled.commands and controlled.loads == 0


@pytest.mark.parametrize('problem', ['split', 'memories', 'partial_coverage', 'duplicate_history', 'singleton_batch', 'quantized_teacher'])
def test_validation_spec_and_teacher_refuse_invalid_development_inputs(tmp_path, monkeypatch, problem):
    pipeline, config, original = setup_evaluation(tmp_path, monkeypatch, complete=False)
    controlled = ControlledEvaluation(original, monkeypatch);value = json.loads(config.read_text())
    if problem == 'split':value['split'] = 'test'
    elif problem == 'memories':value['memories'] = ['normal']
    elif problem == 'partial_coverage':value['validation_examples'] = 8
    elif problem == 'singleton_batch':value['batch_size'] = 5
    elif problem == 'quantized_teacher':
        path = Path(value['teacher_config']);teacher = json.loads(path.read_text());teacher['search_consolidator']['quantization'] = 'q4';atomic_json(path, teacher)
    else:
        path = Path(value['data']) / 'validation.jsonl';rows = load_jsonl(path);rows[1]['feature_key'] = rows[0]['feature_key'];write_jsonl(path, rows)
        token_path = Path(value['token_preflight']);token = json.loads(token_path.read_text());token['inputs'][str(path)] = artifact(path);atomic_json(token_path, token)
    atomic_json(config, value)
    with pytest.raises(ValueError):evaluation.run_evaluation(pipeline, 'owned-sft', config, tmp_path / 'invalid')
    assert not controlled.commands and controlled.loads == 0


def test_actual_selected_checkpoint_input_contract_is_checked_even_with_rebound_proof_hashes(tmp_path, monkeypatch):
    pipeline, config, original = setup_evaluation(tmp_path, monkeypatch);controlled = ControlledEvaluation(original, monkeypatch)
    path = pipeline / 'sft/adapter.pt';checkpoint = torch.load(path, weights_only=True);checkpoint['input_hashes'] = {};torch.save(checkpoint, path)
    trained_path = pipeline / 'sft/manifest.json';trained = json.loads(trained_path.read_text());trained['outputs'][str(path)] = artifact(path);atomic_json(trained_path, trained)
    whole_path = pipeline / 'manifest.json';whole = json.loads(whole_path.read_text());whole['outputs'][str(path)] = artifact(path);whole['inputs'][str(trained_path)] = artifact(trained_path);atomic_json(whole_path, whole)
    with pytest.raises(ValueError, match='actual selected'):
        evaluation.run_evaluation(pipeline, 'owned-sft', config, tmp_path / 'refused')
    assert not controlled.commands and controlled.loads == 0


@pytest.mark.parametrize('failure', ['changed_judged_raw', 'false_metrics', 'wrong_memory', 'foreign_response', 'foreign_reviewer_input', 'short_batch'])
def test_invalid_phase_outputs_fail_and_preserve_partial_evidence(tmp_path, monkeypatch, failure):
    pipeline, config, original = setup_evaluation(tmp_path, monkeypatch);controlled = ControlledEvaluation(original, monkeypatch);controlled.failure = failure
    output = tmp_path / 'failure'
    with pytest.raises(ValueError):evaluation.run_evaluation(pipeline, 'owned-sft', config, output)
    assert json.loads((output / 'state.json').read_text())['status'] == 'failed'
    assert not (output / 'manifest.json').exists()
    assert (output / 'commands.jsonl').exists()
    if failure in ['changed_judged_raw', 'false_metrics', 'wrong_memory', 'short_batch']:
        assert not any('xqgeneral.local_teacher' in c for c in controlled.commands)


def test_incomplete_raw_run_is_preserved_and_cannot_be_overwritten_on_resume(tmp_path, monkeypatch):
    pipeline, config, original = setup_evaluation(tmp_path, monkeypatch);controlled = ControlledEvaluation(original, monkeypatch)
    output = tmp_path / 'incomplete';controlled.failure = 'false_metrics'
    with pytest.raises(ValueError):evaluation.run_evaluation(pipeline, 'owned-sft', config, output)
    (output / 'normal/manifest.json').rename(output / 'normal/preserved-failed-manifest.json')
    count = len(controlled.commands);controlled.failure = None
    with pytest.raises(ValueError, match='Preserve incomplete raw'):
        evaluation.run_evaluation(pipeline, 'owned-sft', config, output, resume=True)
    assert len(controlled.commands) == count and (output / 'normal/raw-predictions.jsonl').exists()
