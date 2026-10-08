import json
from pathlib import Path
import random
import sys

import pytest
import torch

from xqgeneral.board_tokens import PAIRS
from xqgeneral.bridge import GatedBridge
from xqgeneral.curriculum_data import STAGES, dynamic_tasks, static_tasks
from xqgeneral.evaluate_qa import balanced_rows, qa_summary
from xqgeneral.evidence import atomic_json, code_identity, digest, history_key, manifest, write_jsonl
from xqgeneral.foundation_handoff import export_curriculum
from xqgeneral.gated_curriculum import raw_qa_gate, training_config
from xqgeneral.rules import START_FEN, replay
from xqgeneral.selfplay_grounding import question_variant


def fixture_rows():
    history = replay(START_FEN, ['h2e2', 'h7e7']);future = ['b0c2']
    rows = []
    for stage in STAGES:
        is_future = 'future' in stage
        fen = replay(history[-1], future)[-1] if is_future else history[-1]
        for i in range(3):
            tasks = static_tasks(fen, random.Random(i)) if stage.startswith('static') else dynamic_tasks(fen, random.Random(i))
            for task, question, answer, query in tasks:
                if is_future:question = '依次走 b0c2 后，' + question
                rows.append({'id': f'{stage}/{task}/{i}', 'game_id': 'controlled-validation-game',
                    'split': 'validation', 'stage': stage, 'task_type': task, 'query': query,
                    'initial_fen': START_FEN, 'moves': ['h2e2', 'h7e7'], 'history': history,
                    'fen': history[-1], 'feature_key': history_key(history),
                    'future_moves': future if is_future else [], 'question': question, 'answer': answer})
    return rows


def completed_curriculum(tmp_path, *, wrong_gold=False, wrong_weight=None):
    """Controlled tensor/QA artifact integration; no model was actually trained."""
    root = tmp_path / 'curriculum';data = tmp_path / 'data';model = tmp_path / 'model'
    rows = fixture_rows()
    if wrong_gold:rows[0]['answer'] = 'incorrect-native-target'
    write_jsonl(data / 'validation.jsonl', rows);(data / 'train.jsonl').write_text('controlled-training-file\n')
    features = tmp_path / 'features.pt';features.write_bytes(b'controlled-feature-identity')
    atomic_json(model / 'config.json', {'hidden_size': 6, 'num_hidden_layers': 4})
    recipe = json.loads(Path('configs/foundation-human-engine-clean-v2.json').read_text())
    recipe.update(model_path=str(model), model_revision='controlled-base-revision', output=str(root),
        data_path=str(data), feature_path=str(features), decoder_bridge_positions=[0, 2],
        expert_feature_depths=[0, 1], bridge_width=6, ddp_world_size=1, batch_size=4, micro_batch_size=1)
    recipe['raw_qa_gates'].update(per_task=3, seed=13)
    for budget in recipe['course_budgets']:budget.update(steps=16, minimum_steps=8, qa_every=8)
    recipe_path = tmp_path / 'recipe.json';atomic_json(recipe_path, recipe)
    code = code_identity();parent = None;results = []
    for index, stage in enumerate(STAGES):
        config = training_config(recipe, index, root, parent);stage_root = root / stage
        candidate = stage_root / 'candidates' / 'step-8';checkpoint = candidate / 'adapter.pt'
        candidate.mkdir(parents=True)
        config_path = stage_root / 'training.config.json';atomic_json(config_path, config)
        bridge = GatedBridge(6, 512, 6)
        state = {f'bridges.{i}.{n}': torch.full_like(p, .125 * (index + 1))
                 for i in range(2) for n, p in bridge.named_parameters()}
        state.update({n: torch.full((len(PAIRS), 6), .25 * (index + 1)) for n in
            ['base.model.embed_tokens.board_weight', 'base.lm_head.board_weight']})
        if index == 3:
            if wrong_weight == 'incomplete_bridge':state.pop('bridges.1.q.weight')
            elif wrong_weight == 'missing_board_embedding':state.pop('base.lm_head.board_weight')
            elif wrong_weight == 'wrong_shape':state['bridges.1.q.weight'] = torch.ones((99, 99))
            elif wrong_weight == 'low_precision':state['bridges.1.q.weight'] = state['bridges.1.q.weight'].half()
            elif wrong_weight == 'nonfinite':state['bridges.1.q.weight'].fill_(float('nan'))
            elif wrong_weight == 'unexpected_decoder_weight':state['base.untrained_decoder.weight'] = torch.ones(1)
        inputs = [features, data / 'train.jsonl', data / 'validation.jsonl']
        if parent:inputs.append(parent)
        saved = {'trainable': state, 'config': config, 'selected_step': 8,
                 'input_hashes': {str(p): digest(p) for p in inputs}, 'code': code}
        torch.save(saved, checkpoint)
        source = candidate / 'source-checkpoint.pt';torch.save(dict(saved, step=8, optimizer={}), source)
        snapshot = candidate / 'snapshot.json';atomic_json(snapshot, {
            'actual_step': 8, 'atomic_source_inode_pinned': True, 'source_sha256': digest(source),
            'adapter_sha256': digest(checkpoint), 'optimizer_state_in_adapter': False,
            'training_config': config, 'training_input_hashes': saved['input_hashes'], 'training_code': code})
        counts, predictions = {}, []
        for row in balanced_rows(rows, 3, 13, STAGES[:index + 1]):
            key = row['stage'], row['task_type'];variant = counts.get(key, 0) % 3;counts[key] = counts.get(key, 0) + 1
            predictions.append({k: row[k] for k in ['id', 'game_id', 'stage', 'task_type']} | {
                'question': question_variant(row, variant), 'question_variant': variant,
                'expected': row['answer'], 'generated': row['answer'], 'correct': True})
        qa = candidate / 'qa';write_jsonl(qa / 'predictions.jsonl', predictions)
        metrics = dict(qa_summary(predictions), raw_generation=True, oracle_used=False, split='validation')
        atomic_json(qa / 'metrics.json', metrics)
        args = {'checkpoint': str(checkpoint), 'data': str(data), 'features': str(features),
                'split': 'validation', 'memory': 'normal', 'stages': list(STAGES[:index + 1]),
                'per_task': 3, 'seed': 13, 'question_formats': 3}
        atomic_json(qa / 'manifest.json', manifest('balanced_board_qa', args,
            [checkpoint, data / 'validation.jsonl', features],
            [qa / 'predictions.jsonl', qa / 'metrics.json'], metrics, code=code))
        gate = raw_qa_gate(predictions, STAGES[:index + 1], recipe['raw_qa_gates']['targets'], 3)
        atomic_json(candidate / 'gate.json', gate)
        result = {'stage': stage, 'actual_steps': 8, 'global_batch_size': 4, 'sample_presentations': 32,
            'gate': gate, 'selected_checkpoint': str(checkpoint), 'selected_checkpoint_sha256': digest(checkpoint),
            'next_course_permitted': True, 'optimizer_reset_between_courses': True, 'independent_test_used': False}
        atomic_json(stage_root / 'manifest.json', manifest('raw_qa_gated_foundation_course', config,
            [recipe_path, config_path, snapshot, qa / 'manifest.json'], [checkpoint, candidate / 'gate.json'], result, code=code))
        results.append(result);parent = checkpoint
    atomic_json(root / 'manifest.json', manifest('four_course_raw_qa_gated_curriculum', recipe,
        [recipe_path, *[root / s / 'manifest.json' for s in STAGES]], [parent],
        {'courses': results, 'all_raw_qa_gates_passed': True, 'independent_test_used': False}, code=code))
    return root, parent


def test_handoff_exports_exact_final_weights_and_format_consumed_by_sft(tmp_path, monkeypatch):
    from xqgeneral import sft
    root, selected = completed_curriculum(tmp_path);output = tmp_path / 'handoff'
    result = export_curriculum(root, output)
    assert (output / 'adapter.pt').samefile(selected) and digest(output / 'adapter.pt') == digest(selected)
    assert result['all_four_ordered_raw_qa_gates_recomputed'] and result['exported_weights_unchanged']
    assert result['sampled_native_histories_and_gold_answers_recomputed'] == 66
    assert not result['independent_test_answers_used'] and not result['model_weights_loaded_on_gpu_or_sft_started']
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['status'] == 'complete' and proof['kind'] == 'raw_qa_gated_foundation_handoff'
    assert proof['outputs'][str(output / 'adapter.pt')]['sha256'] == digest(selected)
    explanation = tmp_path / 'explanations';write_jsonl(explanation / 'train.jsonl', [{'stage': 'explanation'}])
    recipe_path = tmp_path / 'sft.json';atomic_json(recipe_path, {'data_path': str(explanation),
        'stages': ['explanation'], 'mixture': {'explanation': 1}, 'epochs': 1, 'max_steps': 8,
        'batch_size': 1, 'min_steps': 1, 'decoder_training': 'full', 'require_clean_foundation_handoff': True})
    calls = [];monkeypatch.setattr(sft.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(sys, 'argv', ['sft', '--recipe', str(recipe_path), '--init', str(output / 'adapter.pt'), '--output', str(tmp_path / 'sft')])
    sft.main()
    config = json.loads((tmp_path / 'sft.config.json').read_text())
    assert config['init_from'] == str(output / 'adapter.pt') and config['decoder_training'] == 'full'
    assert config['model_revision'] == 'controlled-base-revision' and config['decoder_bridge_positions'] == [0, 2]
    assert len(calls) == 1 and calls[0][1:3] == ['-m', 'xqgeneral.train']
    with pytest.raises(FileExistsError):export_curriculum(root, output)


@pytest.mark.parametrize('problem', ['missing', 'partial', 'failed_gate', 'test_selection', 'wrong_order', 'inherited_old', 'literal_board'])
def test_incomplete_or_wrong_foundation_never_creates_handoff(tmp_path, problem):
    root, _ = completed_curriculum(tmp_path);path = root / 'manifest.json';doc = json.loads(path.read_text())
    if problem == 'missing':path.unlink()
    else:
        if problem == 'partial':doc['verification']['courses'].pop()
        elif problem == 'failed_gate':doc['verification']['all_raw_qa_gates_passed'] = False
        elif problem == 'test_selection':doc['verification']['independent_test_used'] = True
        elif problem == 'wrong_order':doc['verification']['courses'].reverse()
        elif problem == 'inherited_old':doc['config']['init_from'] = 'old-inherited-course.pt'
        elif problem == 'literal_board':doc['config']['board_text'] = 'dictionary'
        atomic_json(path, doc)
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError):export_curriculum(root, output)
    assert not output.exists()


@pytest.mark.parametrize('artifact', ['prediction', 'training_data', 'feature_cache', 'snapshot', 'candidate', 'base_config'])
def test_changed_completed_inputs_or_weights_are_rejected_before_export(tmp_path, artifact):
    root, selected = completed_curriculum(tmp_path);doc = json.loads((root / 'manifest.json').read_text())
    paths = {'prediction': selected.parent / 'qa' / 'predictions.jsonl',
        'training_data': Path(doc['config']['data_path']) / 'train.jsonl',
        'feature_cache': Path(doc['config']['feature_path']), 'snapshot': selected.parent / 'snapshot.json',
        'candidate': selected, 'base_config': Path(doc['config']['model_path']) / 'config.json'}
    with paths[artifact].open('ab') as handle:handle.write(b'changed')
    output = tmp_path / 'not-created'
    with pytest.raises((ValueError, json.JSONDecodeError)):export_curriculum(root, output)
    assert not output.exists()


@pytest.mark.parametrize('problem', ['native_target', 'incomplete_bridge', 'missing_board_embedding',
                                   'wrong_shape', 'low_precision', 'nonfinite', 'unexpected_decoder_weight'])
def test_self_consistent_declared_proofs_cannot_hide_wrong_native_answers_or_incomplete_state(tmp_path, problem):
    root, _ = completed_curriculum(tmp_path, wrong_gold=problem == 'native_target', wrong_weight=problem)
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='native rules|every bridge|incompatible shapes'):export_curriculum(root, output)
    assert not output.exists()


def test_distinct_training_and_feature_artifacts_are_freshly_hashed_once(tmp_path, monkeypatch):
    from collections import Counter
    from xqgeneral import foundation_handoff
    root, _ = completed_curriculum(tmp_path);calls = Counter()
    original = foundation_handoff.digest
    def counted(path):
        calls[str(path)] += 1
        return original(path)
    monkeypatch.setattr(foundation_handoff, 'digest', counted)
    export_curriculum(root, tmp_path / 'handoff')
    assert calls[str(tmp_path / 'features.pt')] == 1
    assert calls[str(tmp_path / 'data' / 'train.jsonl')] == 1
    assert calls[str(tmp_path / 'data' / 'validation.jsonl')] == 1


def test_evidence_changed_during_state_validation_never_creates_handoff(tmp_path, monkeypatch):
    from xqgeneral import foundation_handoff
    root, _ = completed_curriculum(tmp_path);original = foundation_handoff.foundation_state_summary
    def changed(checkpoint, base_config):
        with (tmp_path / 'data' / 'validation.jsonl').open('ab') as handle:handle.write(b'changed-during-verification')
        return original(checkpoint, base_config)
    monkeypatch.setattr(foundation_handoff, 'foundation_state_summary', changed)
    with pytest.raises(ValueError, match='changed during verification'):
        export_curriculum(root, tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('problem', ['failed_course', 'wrong_order', 'missing_completion', 'tampered_config',
                                   'model_revision', 'literal_board', 'expert_depths', 'bridge_width'])
def test_sft_rejects_broken_handoffs_and_architecture_changes_before_starting_training(tmp_path, monkeypatch, problem):
    from xqgeneral import sft
    root, _ = completed_curriculum(tmp_path);output = tmp_path / 'handoff';export_curriculum(root, output)
    manifest_path = output / 'manifest.json';doc = json.loads(manifest_path.read_text())
    if problem == 'failed_course':doc['verification']['courses'][-1]['gate']['passed'] = False
    elif problem == 'wrong_order':doc['verification']['courses'].reverse()
    elif problem == 'missing_completion':doc['verification'].pop('four_course_handoff_export_completed')
    if problem in ('failed_course', 'wrong_order', 'missing_completion'):atomic_json(manifest_path, doc)
    if problem == 'tampered_config':
        config_path = output / 'config.json';saved = json.loads(config_path.read_text())
        atomic_json(config_path, dict(saved, bridge_width=12))
    recipe = {'stages': ['explanation'], 'data_path': str(tmp_path / 'unused'), 'epochs': 1,
              'max_steps': 8, 'batch_size': 1, 'min_steps': 1, 'mixture': {'explanation': 1}}
    if problem == 'model_revision':recipe['model_revision'] = 'different-base-revision'
    elif problem == 'literal_board':recipe['board_text'] = 'dictionary'
    elif problem == 'expert_depths':recipe['expert_feature_depths'] = [1, 0]
    elif problem == 'bridge_width':recipe['bridge_width'] = 12
    path = tmp_path / 'sft.json';atomic_json(path, recipe)
    def forbidden(*args, **kwargs):raise AssertionError('Training must not start from an invalid handoff')
    monkeypatch.setattr(sft.subprocess, 'run', forbidden)
    monkeypatch.setattr(sys, 'argv', ['sft', '--recipe', str(path), '--init', str(output / 'adapter.pt'), '--output', str(tmp_path / 'not-created')])
    with pytest.raises(ValueError, match='four ordered|configuration changed|preserve the selected'):
        sft.main()
    assert not (tmp_path / 'not-created').exists() and not (tmp_path / 'not-created.config.json').exists()
