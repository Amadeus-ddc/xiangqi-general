"""Native replay/selection controls; constructed sources do not prove model quality."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

import test_oracle_policy_corpus as corpus_control
import test_distillation_joint_replay as joint_control
from test_paper_curriculum import source_fixture
from xqgeneral import auxiliary_search_replay as replay, distillation_preflight, distillation_training
from xqgeneral.course_tasks import paper_records
from xqgeneral.distillation_validation import AUXILIARY_VALIDATION_PROFILE, joint_validation_coverage
from xqgeneral.evidence import atomic_json, digest, iter_jsonl, load_jsonl, manifest, write_jsonl
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.oracle_policy_corpus import ALL_STAGES, KIND, PROFILE
from xqgeneral.oracle_policy_data import _numeric_after_isolation
from xqgeneral.policy_data import QUESTION
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.rules import START_FEN, legal_moves, play
from xqgeneral.search_distillation import NUMERIC_CHILD_QUESTION
from xqgeneral.training_index import IndexedTrainingRows, build_index


def dataset(tmp_path):
    _, _, roots = source_fixture(tmp_path / 'qa-sources')
    rows = []
    for row in roots:
        if row['split'] != 'test':
            rows.extend(paper_records(row, random.Random(71)))
    reds = sorted(legal_moves(START_FEN))
    for split, red_values in [('train', reds[:3]), ('validation', reds[-2:])]:
        for red in red_values:
            black = sorted(legal_moves(play(START_FEN, red)))[-1]
            context = corpus_control.context([red, black], split)
            target, _ = _numeric_after_isolation(corpus_control.query(context, 1000000))
            assert target is not None
            rows.extend([
                corpus_control.explanation(context),
                dict(context, id='numeric-' + context['id'], stage='engine_numeric',
                     task_type='move_PV_candidates_evaluation_v1', question=NUMERIC_CHILD_QUESTION,
                     answer=json.dumps(target)),
                dict(context, id='move-' + context['id'], stage='move_quality', task_type='best_move',
                     question=QUESTION, answer=target['move']),
            ])
    data = tmp_path / 'completed-corpus'
    for split in ['train', 'validation', 'test']:
        write_jsonl(data / (split + '.jsonl'), [r for r in rows if r['split'] == split])
    required = ['unchanged_source_row_bytes_retained', 'original_validation_row_bytes_preserved',
        'upstream_complete_native_history_proofs_reused', 'all_supervised_positions_and_mirrors_isolated',
        'no_new_policy_validation_target_enters_other_protected_positions']
    result = dict.fromkeys(required, True)
    result.update({split + '_rows_by_stage': dict(Counter(r['stage'] for r in rows if r['split'] == split))
                   for split in ['train', 'validation']})
    result.update(test_rows=0, final_independent_test_rows_parsed_or_tokenized=0)
    atomic_json(data / 'manifest.json', manifest(KIND, dict(profile=PROFILE, constructed_fixture=True),
        outputs=[data / (s + '.jsonl') for s in ['train', 'validation', 'test']], verification=result))
    return data


def template():
    return json.loads(Path('configs/search-distillation-sft-three-gpu-with-auxiliary-v1.json').read_text())


def budget():
    value, token = joint_control.joint_budget(train=31, old=40)
    value = template()
    value['replay_data_paths'] = token['config']['replay_data_paths']
    token['config']['replay_profile'] = replay.PROFILE
    result = token['verification']['course_replay']
    for split in ['train', 'validation']:
        result['by_split_and_stage'][split].update(engine_numeric=40 if split == 'train' else 5,
                                                  move_quality=40 if split == 'train' else 6)
    result['train_validation_rows_tokenized'] += 40 * 2 + 11
    return value, token


def test_native_subset_keeps_selected_training_bytes_and_every_original_validation_row(tmp_path):
    data = dataset(tmp_path)
    output = tmp_path / 'replay'
    result = replay.prepare(data, output, train_per_stage=2)
    assert result['by_split']['train'] == 12
    assert set(result['by_split_and_stage']['train']) == set(replay.STAGES)
    expected = b''.join(raw for raw in (data / 'validation.jsonl').read_bytes().splitlines(keepends=True)
                        if json.loads(raw)['stage'] != 'explanation')
    assert (output / 'validation.jsonl').read_bytes() == expected
    originals = {json.loads(raw)['id']: raw for raw in (data / 'train.jsonl').read_bytes().splitlines(keepends=True)}
    assert all(originals[json.loads(raw)['id']] == raw
               for raw in (output / 'train.jsonl').read_bytes().splitlines(keepends=True))
    before = corpus_control.snapshot(output)
    assert replay.prepare(data, output, train_per_stage=2, resume=True) == result
    assert corpus_control.snapshot(output) == before
    rows, loaded = distillation_preflight.course_replay_sources([output], BoundInputs(), profile=replay.PROFILE)
    assert len(rows) == result['by_split']['train'] + result['by_split']['validation']
    assert loaded['by_split_and_stage'] == result['by_split_and_stage']
    with pytest.raises(ValueError):
        distillation_preflight.course_replay_sources([output], BoundInputs())


@pytest.mark.parametrize('damage', ['omit_train', 'omit_validation', 'reorder_validation', 'change_question',
                                  'change_answer', 'change_stage', 'test', 'claim'])
def test_rehashed_replay_cannot_replace_native_source_selection(tmp_path, damage):
    data = dataset(tmp_path); out = tmp_path / 'subset'; replay.prepare(data, out, train_per_stage=2)
    split = 'validation' if damage in ['omit_validation', 'reorder_validation'] else 'train'
    path = out / (split + '.jsonl'); rows = load_jsonl(path)
    if damage.startswith('omit_'): rows.pop()
    elif damage == 'reorder_validation': rows.reverse()
    elif damage == 'change_question': rows[0]['question'] += '改变'
    elif damage == 'change_answer': rows[0]['answer'] = '非法答案'
    elif damage == 'change_stage': rows[0]['stage'] = 'explanation'
    elif damage == 'test': rows[0]['split'] = 'test'
    write_jsonl(path, rows)
    proof = json.loads((out / 'manifest.json').read_text())
    proof['outputs'][str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    if damage == 'claim': proof['verification']['by_split']['train'] += 1
    atomic_json(out / 'manifest.json', proof)
    with pytest.raises(ValueError): replay.checked_subset(out, BoundInputs())


@pytest.mark.parametrize('quota,seed', [(0, 1), (True, 1), (1, -1), (1, False)])
def test_invalid_sampling_contract_does_not_create_output(tmp_path, quota, seed):
    output = tmp_path / 'unused'
    with pytest.raises(ValueError): replay.prepare(tmp_path / 'unused-source', output,
                                                  train_per_stage=quota, seed=seed)
    assert not output.exists()


def test_seven_stage_search_budget_derives_all_declared_subset_rows_and_validates_every_stage(tmp_path):
    value, token = budget()
    compiled, actual = distillation_training.compile_recipe(value, token, tmp_path)
    expected = dict(explanation=31, **dict.fromkeys(replay.STAGES, 40))
    assert actual['finite_mixture']['selected_rows_by_stage'] == expected
    assert actual['rows_per_complete_pass'] == 271
    assert compiled['validation_rows_by_stage'] == dict(explanation=384,
        **dict.fromkeys(replay.COURSES, 32), engine_numeric=5, move_quality=6)
    assert compiled['validation_examples'] == 523
    assert actual['auxiliary_policy_replay_included']
    assert not actual['auxiliary_replay_counts_toward_new_search_label_scale']
    rows = [dict(id=f'{stage}-{i}', stage=stage, feature_key=f'{stage}-{i}', split='train', answer='controlled')
            for stage, count in expected.items() for i in range(count)]
    path = tmp_path / 'native-rows.jsonl'; write_jsonl(path, rows); build_index([path], tmp_path / 'index')
    indexed = IndexedTrainingRows(tmp_path / 'index', [path], compiled['mixture'])
    try:
        finite = prepare_finite_epochs(indexed, compiled['mixture'], dict(compiled, steps=actual['max_updates']), 3)
        observed = Counter()
        for step in range(actual['updates_per_complete_pass']):
            for rank in range(3):
                indices, _ = finite.batch_indices(step, rank)
                observed.update(indexed.row_metadata(i)['id'] for i in indices)
        assert set(observed) == {r['id'] for r in rows}
        assert sum(observed.values()) == 273
    finally: indexed.close()


@pytest.mark.parametrize('damage', ['token_profile', 'template_profile', 'drop_numeric', 'validation_stage',
                                  'source_profile', 'loss', 'token_count', 'four_workers'])
def test_auxiliary_replay_cannot_silently_revert_to_five_stage_training(tmp_path, damage):
    value, token = budget()
    if damage == 'token_profile': token['config'].pop('replay_profile')
    elif damage == 'template_profile': value.pop('replay_profile')
    elif damage == 'drop_numeric': token['verification']['course_replay']['by_split_and_stage']['train'].pop('engine_numeric')
    elif damage == 'validation_stage': value['validation_stages'].pop()
    elif damage == 'source_profile': value.pop('source_coverage_profile')
    elif damage == 'loss': value['loss_normalization'] = 'example'
    elif damage == 'token_count': token['verification']['course_replay']['train_validation_rows_tokenized'] -= 1
    else: value.update(ddp_world_size=4, batch_size=32, search_topology_profile='full_decoder_search_four_workers_v1')
    with pytest.raises(ValueError): distillation_training.compile_recipe(value, token, tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def test_full_seven_stage_native_validation_ids_reject_omitted_policy_rows():
    rows = [dict(id=s, stage=s, split='validation') for s in ALL_STAGES]
    expected = dict.fromkeys(ALL_STAGES, 1)
    result = joint_validation_coverage(rows, list(ALL_STAGES), expected, profile=AUXILIARY_VALIDATION_PROFILE)
    assert result['profile'] == AUXILIARY_VALIDATION_PROFILE and result['validation_rows'] == 7
    with pytest.raises(ValueError):
        joint_validation_coverage(rows, list(ALL_STAGES)[:-1], expected, profile=AUXILIARY_VALIDATION_PROFILE)


def test_native_search_token_gate_covers_six_task_subset_with_original_features_and_raw_rows(tmp_path, monkeypatch):
    from test_feature_store import block
    from xqgeneral.feature_store import FeatureStoreWriter
    args, _, _, originals, tokenized = joint_control.native_replay(tmp_path, monkeypatch)
    data = dataset(tmp_path / 'auxiliary-source')
    subset = tmp_path / 'six-task-replay'
    selected = replay.prepare(data, subset, train_per_stage=2)
    rows = [*load_jsonl(subset / 'train.jsonl'), *load_jsonl(subset / 'validation.jsonl')]
    old_cache = Path(args[3]); original_cache = Path(json.loads(old_cache.read_text())['config']['base_cache'])
    writer = FeatureStoreWriter(tmp_path / 'six-task-features', [0, 3], 'a' * 64,
        config=dict(base_cache=str(original_cache), base_proof=str(original_cache)))
    writer.add_existing(original_cache, original_cache)
    existing = set(json.loads(Path(json.loads(original_cache.read_text())['catalog']['keys']).read_text()))
    earlier = set(json.loads(Path(json.loads(old_cache.read_text())['catalog']['keys']).read_text()))
    writer.append(block(sorted(({r['feature_key'] for r in rows} | earlier) - existing), width=512)); writer.finish()
    result = distillation_preflight.preflight(*[*args[:3], writer.output / 'manifest.json', args[4]],
        tmp_path / 'actual-token-gate', workers=1, replay_data_paths=[subset], replay_profile=replay.PROFILE)
    assert result['course_replay']['by_split_and_stage'] == selected['by_split_and_stage']
    assert set(tokenized) == {r['id'] for r in [*originals, *rows] if r['split'] != 'test'}
    assert result['independent_test_rows_tokenized'] == 0


@pytest.mark.parametrize('missing_policy_ledger', [False, True])
def test_native_three_worker_search_round_consumes_seven_stage_validation_and_reopens(tmp_path, monkeypatch,
                                                                                  missing_policy_ledger):
    import test_distillation_training as training_control
    from test_three_gpu_search_distillation import template as three_template
    from xqgeneral.distillation_preflight import completed_parent
    from xqgeneral.training import select_validation_rows
    monkeypatch.setattr(training_control, 'template', three_template)
    source, recipe_path, token_path, runner, raw = joint_control.controlled_joint_round(tmp_path, monkeypatch)
    token = json.loads(token_path.read_text())
    for split, count in [('train', 4), ('validation', 2)]:
        rows = load_jsonl(raw / (split + '.jsonl'))
        rows += [dict(id=f'{stage}-{split}-{i}', feature_key=f'policy-{stage}-{split}-{i}',
                      stage=stage, split=split, answer='controlled; no neural quality claim')
                 for stage in ['engine_numeric', 'move_quality'] for i in range(count)]
        write_jsonl(raw / (split + '.jsonl'), rows)
        for stage in ['engine_numeric', 'move_quality']:
            token['verification']['course_replay']['by_split_and_stage'][split][stage] = count
    token['verification']['course_replay']['train_validation_rows_tokenized'] += 12
    token['config']['replay_profile'] = replay.PROFILE
    for name in ['train.jsonl', 'validation.jsonl']:
        path = raw / name
        token['inputs'][str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    value = dict(template(), replay_data_paths=[str(raw)], max_steps=12, max_finite_passes=2, eval_every=2)
    atomic_json(recipe_path, value); atomic_json(token_path, token)
    def run(command, **kwargs):
        result = runner(command, **kwargs)
        if 'xqgeneral.sft' in command:
            root = Path(command[command.index('--output') + 1]); config = json.loads((root / 'config.json').read_text())
            indexed = IndexedTrainingRows(config['row_index_paths']['validation'],
                [Path(p) / 'validation.jsonl' for p in [config['data_path'], *config['replay_data_paths']]],
                config['mixture'], split='validation')
            try:
                ids = [r['id'] for r in select_validation_rows(indexed, config['validation_stages'],
                       config['seed'], config['validation_examples'])]
            finally: indexed.close()
            if missing_policy_ledger: ids.pop()
            atomic_json(root / 'validation_ids.json', ids)
        return result
    monkeypatch.setattr(distillation_training.subprocess, 'run', run)
    output = tmp_path / 'seven-stage-search-round'
    if missing_policy_ledger:
        with pytest.raises(ValueError): distillation_training.run_round(source, recipe_path, token_path, output)
        assert not (output / 'manifest.json').exists()
    else:
        result = distillation_training.run_round(source, recipe_path, token_path, output)
        assert result['joint_validation_coverage']['profile'] == AUXILIARY_VALIDATION_PROFILE
        assert result['joint_validation_coverage']['validation_rows'] == 28
        assert set(result['selected_training_rows_by_stage']) == set(ALL_STAGES)
        completed_parent(output / 'sft/adapter.pt', output / 'manifest.json')
        assert distillation_training.run_round(source, recipe_path, token_path, output, resume=True) == result
        assert len(runner.jobs) == 4


@pytest.mark.parametrize('module,arguments,target', [
    (distillation_preflight, ['--init', 'parent.pt', '--parent-completion', 'parent.json', '--data', 'labels',
        '--features', 'features.json', '--foundation-preflight', 'foundation.json', '--output', 'tokens'], 'preflight'),
    (distillation_training, ['--init', 'parent.pt', '--recipe', 'recipe.json',
        '--token-preflight', 'tokens.json', '--output', 'round'], 'run_round'),
])
def test_native_search_CLI_activates_explicit_verified_input_receipt_session(monkeypatch, module, arguments, target):
    from contextlib import contextmanager
    import sys
    from xqgeneral import verified_inputs
    active = []
    @contextmanager
    def session():
        active.append(True)
        try: yield
        finally: active.pop()
    calls = []
    def execute(*args, **kwargs):
        assert active == [True]
        calls.append((args, kwargs))
        return dict(controlled_entry_only=True)
    monkeypatch.setattr(verified_inputs, 'input_session_from_environment', session)
    monkeypatch.setattr(module, target, execute)
    monkeypatch.setattr(sys, 'argv', ['controlled-search-cli', *arguments])
    module.main()
    assert len(calls) == 1 and active == []


def native_draw_row(stage='engine_numeric'):
    from xqgeneral.rules import adjudicate
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    context = corpus_control.context(cycle)
    choices = sorted(legal_moves(context['fen']))
    target = dict(move=cycle[0], pv=cycle, candidates=[cycle[0], *[m for m in choices if m != cycle[0]][:2]],
                  evaluation=dict(type='cp', value=0, perspective='side_to_move'))
    assert not adjudicate(context['initial_fen'], context['moves'])['ended']
    ending = adjudicate(context['initial_fen'], [*context['moves'], *cycle])
    assert ending['ended'] and ending['winner'] is None
    return dict(context, id='constructed-draw-' + stage, stage=stage,
        task_type='move_PV_candidates_evaluation_v1' if stage == 'engine_numeric' else 'best_move',
        question=NUMERIC_CHILD_QUESTION if stage == 'engine_numeric' else QUESTION,
        answer=json.dumps(target) if stage == 'engine_numeric' else target['move'],
        future_moves=cycle, target_line_termination=ending)


@pytest.mark.parametrize('stage', ['engine_numeric', 'move_quality'])
def test_native_draw_policy_target_is_not_a_paper_future_board_question(stage):
    from xqgeneral.oracle_policy_evaluation import grade, prompt_record
    from xqgeneral.training import messages
    row = native_draw_row(stage)
    assert grade(row, row['answer'])['contract_valid']
    assert messages(row) == messages(prompt_record(row))
    assert replay.native_label(row) == row['feature_key']


def test_course_future_board_draw_guard_still_rejects_the_same_ending():
    row = dict(native_draw_row(), stage='static_future', task_type='locate',
               task_profile=corpus_control.corpus.PAPER_PROFILE, query=dict(symbol='H'))
    with pytest.raises(ValueError, match='Paper future ended by history rules'):
        replay.native_label(row)


def test_numeric_PV_cannot_continue_after_a_native_draw():
    row = native_draw_row(); target = json.loads(row['answer']); target['pv'] += ['a0a1']
    row['answer'] = json.dumps(target); row['future_moves'] = target['pv']
    with pytest.raises(ValueError): replay.native_label(row)


@pytest.mark.parametrize('stage', ['engine_numeric', 'move_quality'])
def test_history_terminal_draw_root_guard_remains_for_both_policy_tasks(stage):
    row = native_draw_row(stage); cycle = row['future_moves']
    past = corpus_control.context([*row['moves'], *cycle])
    row.update(past); row['future_moves'] = []
    with pytest.raises(ValueError, match='root ended by history rules'):
        replay.native_label(row)


def test_producer_and_consumer_preserve_a_valid_numeric_draw_row_verbatim(tmp_path):
    data = dataset(tmp_path); row = native_draw_row()
    path = data / 'train.jsonl'; rows = load_jsonl(path); rows.append(row); write_jsonl(path, rows)
    proof = json.loads((data / 'manifest.json').read_text())
    proof['outputs'][str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    proof['verification']['train_rows_by_stage']['engine_numeric'] += 1
    atomic_json(data / 'manifest.json', proof)
    output = tmp_path / 'draw-replay'
    result = replay.prepare(data, output, train_per_stage=2048)
    source_raw = next(raw for raw in path.read_bytes().splitlines(keepends=True) if json.loads(raw)['id'] == row['id'])
    actual_raw = next(raw for raw in (output / 'train.jsonl').read_bytes().splitlines(keepends=True) if json.loads(raw)['id'] == row['id'])
    assert source_raw == actual_raw
    _, checked = replay.checked_subset(output, BoundInputs())
    assert checked == result
