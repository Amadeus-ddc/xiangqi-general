from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_curriculum_selection import imported_recipe, install_curriculum_runner, raw_candidate
from test_foundation_handoff import completed_curriculum, fixture_rows
from xqgeneral.course_position_sampling import (SCARCE_FIRST_PROFILE, PositionCatalog, catalog_inputs,
                                               position_budgets, task_order)
from xqgeneral.course_tasks import PAPER_TASKS
from xqgeneral.curriculum_data import STAGES
from xqgeneral.curriculum_selection import CURRENT_COURSE_METRIC, read_history, selection_decision
from xqgeneral.evidence import atomic_json, code_identity, digest, history_key, write_jsonl
from xqgeneral.feature_store import FeatureStoreWriter, feature_paths
from xqgeneral.foundation_handoff import Artifacts, export_curriculum
from xqgeneral.foundation_preflight import clean_recipe
from xqgeneral.gated_curriculum import training_config
from xqgeneral.rules import START_FEN, replay
from xqgeneral.training import training_input_paths
from xqgeneral.training_index import build_index, index_paths
from xqgeneral.validation_curriculum import run_curriculum, verify_pending


POLICY = {'metric': CURRENT_COURSE_METRIC, 'tie_breaker': 'earliest', 'patience': 3}


def current_gate(current, old, passed=False):
    return {'accuracy': (2 * current + old) / 3, 'by_task': {
        'static_current/piece': {'accuracy': old, 'n': 10},
        'dynamic_current/moves': {'accuracy': current, 'n': 10},
        'dynamic_current/checks': {'accuracy': current, 'n': 10}},
        'courses': {'dynamic_current': {'examples': 20, 'accuracy': current}},
        'validation_only': True, 'raw_generation': True, 'oracle_used': False, 'passed': passed}


def test_current_course_selects_improvement_despite_replay_regression():
    h = [{'step': s, 'gate': current_gate(c, o, True)}
         for s, c, o in [(8, .6, 1.), (16, .8, .2), (24, .7, 1.)]]
    b = {'qa_every': 8, 'steps': 24, 'minimum_steps': 8}
    d = selection_decision(h, POLICY, b, stage='dynamic_current')
    assert d['selected_step'] == 16 and d['selected_accuracy'] == .8
    assert d['stopping_reason'] == 'maximum_budget'
    assert h[2]['gate']['accuracy'] > h[1]['gate']['accuracy']
    assert not selection_decision(h[:1], POLICY, b, stage='dynamic_current')['stop']


def test_post_warmup_best_is_retained_until_minimum_stopping_budget():
    b = {'qa_every': 8, 'steps': 40, 'minimum_steps': 24, 'selection_start_step': 8}
    h = [{'step': s, 'gate': current_gate(c, .9)} for s, c in [(8, .95), (16, .7), (24, .8)]]
    p = dict(POLICY, patience=1)
    assert not selection_decision(h[:2], p, b, stage='dynamic_current')['stop']
    d = selection_decision(h, p, b, stage='dynamic_current')
    assert d['selected_step'] == 8 and d['stopping_reason'] == 'validation_plateau'


def test_current_course_tie_uses_earlier_step_without_replay_tie_breaker():
    h = [{'step': s, 'gate': current_gate(.8, o)} for s, o in [(8, .1), (16, 1.)]]
    d = selection_decision(h, POLICY, {'qa_every': 8, 'steps': 16, 'minimum_steps': 8}, stage='dynamic_current')
    assert d['selected_step'] == 8


@pytest.mark.parametrize('problem', ['stage', 'count', 'accuracy', 'missing_course', 'start_bool', 'start_after_minimum'])
def test_current_course_requires_bound_raw_counts_and_explicit_identity(problem):
    h = [{'step': 8, 'gate': current_gate(.8, .9)}]
    b = {'qa_every': 8, 'steps': 16, 'minimum_steps': 8}
    stage = 'dynamic_current'
    if problem == 'stage':stage = 'static_future'
    elif problem == 'count':h[0]['gate']['courses'][stage]['examples'] = 19
    elif problem == 'accuracy':h[0]['gate']['courses'][stage]['accuracy'] = .81
    elif problem == 'missing_course':h[0]['gate'].pop('courses')
    elif problem == 'start_bool':b['selection_start_step'] = True
    elif problem == 'start_after_minimum':b['selection_start_step'] = 9
    with pytest.raises(ValueError):selection_decision(h, POLICY, b, stage=stage)


def test_fresh_paper_recipe_combines_full_width_finite_token_training_and_raw_selection():
    r = json.loads(Path('configs/foundation-paper-v1.json').read_text())
    clean_recipe(r)
    assert not r.get('init_from') and not r.get('initial_course_import')
    assert r['bridge_architecture'] == 'flamingo_dense' and r['bridge_width'] == 2560
    assert r['loss_normalization'] == 'token' and r['row_index_paths']
    assert r['validation_selection'] == POLICY
    for i, s in enumerate(STAGES):
        c = training_config(r, i, Path(r['output']), None)
        assert c['validation_stages'] == [s]
        assert set(c['mixture']) == set(STAGES[:i + 1])
        assert r['course_budgets'][i]['qa_every'] == 2000


def test_current_course_policy_runs_four_handoffs_and_reports_failed_reference_targets(tmp_path, monkeypatch):
    path, _ = imported_recipe(tmp_path)
    r = json.loads(path.read_text());r['validation_selection'] = dict(POLICY, patience=1)
    atomic_json(path, r);calls = install_curriculum_runner(monkeypatch, r)
    results = run_curriculum(path)
    assert len(calls) == 6
    assert all(v['actual_steps'] == 16 and v['selected_step'] == 8 for v in results)
    for call in calls:
        c = json.loads(Path(call[call.index('--config') + 1]).read_text())
        assert c['validation_stages'] == c['stages']
    assert results == run_curriculum(path, resume=True)
    exported = export_curriculum(r['output'], tmp_path / 'handoff')
    assert exported['all_four_validation_selection_rules_recomputed']
    assert exported['all_reference_targets_passed'] is False


def indexed_candidate(tmp_path):
    original, _ = completed_curriculum(tmp_path)
    r = json.loads((tmp_path / 'recipe.json').read_text())
    rows = fixture_rows()
    write_jsonl(Path(r['data_path']) / 'train.jsonl',
                [dict(v, id='training-' + v['id'], split='train') for v in rows])
    r.update(training_data_profile='author_finite_mixture_epochs_xiangqi_v1', mixture_seed=0,
             row_index_paths={s: str(tmp_path / (s + '-index')) for s in ['train', 'validation']})
    for s, out in r['row_index_paths'].items():build_index([Path(r['data_path']) / (s + '.jsonl')], out, split=s)
    writer = FeatureStoreWriter(tmp_path / 'store', [0, 1], 'a' * 64)
    writer.append({'keys': [rows[0]['feature_key']], 'depths': [0, 1],
                   'features': [torch.zeros((1, 90, 512), dtype=torch.float16) for _ in range(2)],
                   'wdl': torch.zeros((1, 3), dtype=torch.float32)})
    writer.finish();r['feature_path'] = str(tmp_path / 'store/manifest.json')
    c = training_config(r, 0, tmp_path / 'new', None)
    saved = torch.load(original / 'static_current/candidates/step-8/adapter.pt', weights_only=True)
    saved.update(config=c, code=code_identity(), input_hashes={str(p): digest(p) for p in training_input_paths(c)})
    candidates = tmp_path / 'new/static_current/candidates'
    raw_candidate(candidates / 'step-8', saved, 8, r, STAGES[:1])
    return r, c, saved, candidates


def test_indexed_sharded_raw_history_binds_all_storage_and_indexes(tmp_path):
    r, c, saved, candidates = indexed_candidate(tmp_path);artifacts = Artifacts()
    h = read_history(candidates, 8, c, saved['code'], STAGES[:1], r, artifacts)
    verify_pending(artifacts)
    assert h[0]['step'] == 8
    required = {*map(str, feature_paths(r['feature_path'])),
                *[str(p) for d in c['row_index_paths'].values() for p in index_paths(d)]}
    assert required <= set(artifacts.expected)


@pytest.mark.parametrize('artifact', ['shard', 'keys', 'train-index', 'validation-index'])
def test_new_storage_and_index_lineage_cannot_be_omitted(tmp_path, artifact):
    r, c, saved, candidates = indexed_candidate(tmp_path)
    files = feature_paths(r['feature_path'])
    missing = next(p for p in files if str(p).endswith('.pt')) if artifact == 'shard' else (
        next(p for p in files if str(p).endswith('keys.json')) if artifact == 'keys' else
        index_paths(c['row_index_paths'][artifact.removesuffix('-index')])[0])
    saved['input_hashes'].pop(str(missing))
    raw_candidate(candidates / 'step-8', saved, 8, r, STAGES[:1])
    with pytest.raises(ValueError, match='lineage'):
        read_history(candidates, 8, c, saved['code'], STAGES[:1], r, Artifacts())


def test_changed_bound_index_cannot_finish_selection(tmp_path):
    r, c, saved, candidates = indexed_candidate(tmp_path);artifacts = Artifacts()
    read_history(candidates, 8, c, saved['code'], STAGES[:1], r, artifacts)
    with index_paths(c['row_index_paths']['train'])[0].open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError, match='Changed selection evidence'):verify_pending(artifacts)


def test_scarce_dynamic_query_is_reserved_before_generic_moves(tmp_path):
    terminal = '4k4/3R5/4R4/9/9/9/9/9/9/H4K3 b - - 1 1'
    roots = []
    for i, fen in enumerate([terminal, START_FEN]):
        history = replay(fen, [])
        roots.append({'game_id': f'constructed-order-{i}', 'split': 'train', 'initial_fen': fen,
                      'moves': [], 'history': history, 'fen': fen, 'feature_key': history_key(history),
                      'future_moves': [], 'provenance': 'constructed_native_order_probe'})
    catalog = PositionCatalog(tmp_path / 'sources.sqlite')
    try:
        for root, targets in catalog_inputs(roots):catalog.add(root, targets)
        catalog.finish()
        b = position_budgets()
        for split in b:
            for stage in b[split]:b[split][stage] = {t: 1 for t in b[split][stage]}
        old = list(catalog.selections(b, 0, {}));new = list(catalog.selections(b, 0, {}, SCARCE_FIRST_PROFILE))
        old_task = next(t for r, s, t in old if s == 'dynamic_current' and r['fen'] == terminal)
        new_task = next(t for r, s, t in new if s == 'dynamic_current' and r['fen'] == terminal)
        assert old_task == 'moves' and new_task == 'mate'
    finally:catalog.close()


def test_task_order_profile_preserves_task_set_and_is_explicit():
    for s, tasks in task_order(SCARCE_FIRST_PROFILE).items():assert set(tasks) == set(PAPER_TASKS[s])
    assert task_order()['dynamic_current'] == tuple(PAPER_TASKS['dynamic_current'])
    with pytest.raises(ValueError):task_order('unknown')


def test_scarce_first_native_question_generation_and_complete_readback(tmp_path):
    from test_course_position_sampling import fixture, small_budgets
    from xqgeneral.course_sampling import SAMPLING_PROFILE
    from xqgeneral.course_position_sampling import POSITION_PROFILE
    from xqgeneral.paper_curriculum import build, readback
    roots, proof, _ = fixture(tmp_path)
    data = tmp_path / 'questions'
    result = build([roots], [proof], data, color_mirror=False, sampling_profile=SAMPLING_PROFILE,
                   position_sampling_profile=POSITION_PROFILE, task_position_budgets=small_budgets(),
                   task_order_profile=SCARCE_FIRST_PROFILE)
    assert result['task_order_is_local_capacity_adaptation']
    actual = readback(data, tmp_path / 'complete-readback')
    assert actual['questions_regenerated_and_compared'] == result['questions']
    manifest_path = data / 'manifest.json';value = json.loads(manifest_path.read_text())
    value['config'].pop('task_order_profile');atomic_json(manifest_path, value)
    with pytest.raises(ValueError, match='ordering profile'):readback(data, tmp_path / 'reject-removal')
