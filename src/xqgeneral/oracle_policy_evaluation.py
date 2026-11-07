"""Fixed validation diagnostics for raw move and numerical-analysis answers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

from .calibration import evaluation_probability
from .course_tasks import validate_context
from .distillation_preflight import completed_parent
from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest, write_jsonl
from .explanations import continuation_positions, parse_explanation
from .feature_store import FeatureStore, feature_paths
from .model_bundle import _decoder_assets
from .oracle_policy_corpus import ALL_STAGES, KIND as CORPUS_KIND, PROFILE as CORPUS_PROFILE, _basic, _closure, _lines
from .oracle_policy_data import NUMERIC_STAGE
from .recorded_search_inputs import BoundInputs
from .rules import adjudicate, legal_moves
from .sft_preflight import checked_artifacts
from .verified_inputs import input_session_from_environment

INPUT_KIND = 'complete_original_oracle_policy_validation_inputs'
KIND = 'raw_oracle_policy_validation_diagnostics'
STAGES = ('engine_numeric', 'move_quality')
PROMPT_FIELDS = ('id', 'game_id', 'feature_key', 'split', 'initial_fen', 'moves', 'history', 'fen',
                 'stage', 'task_type', 'task_profile', 'question')


def prompt_record(row):
    return {key: row[key] for key in PROMPT_FIELDS if key in row}


def grade(row, raw):
    """Compare to preserved engine references without a new query or answer repair."""
    if row['stage'] not in STAGES or not isinstance(raw, str):
        raise ValueError('Grade only raw strings from the two declared validation tasks')
    legal = legal_moves(row['fen']); errors = []
    if row['stage'] == 'move_quality':
        move = raw.strip()
        formatted = re.fullmatch(r'[a-i][0-9][a-i][0-9]', move) is not None
        valid = formatted and move in legal
        return dict(answer_format_valid=formatted, first_move_legal=valid,
            reference_move_match=valid and move == row['answer'], contract_valid=valid,
            reference_exact=valid and move == row['answer'], raw_exact=raw == row['answer'],
            errors=[] if valid else ['invalid_or_illegal_single_move'])
    expected = json.loads(row['answer'])
    try:
        value = parse_explanation(raw)
    except (ValueError, TypeError):
        return dict(answer_format_valid=False, first_move_legal=False, reference_move_match=False,
            pv_history_legal=False, six_ply_or_terminal=False, candidates_valid=False,
            evaluation_contract_valid=False, contract_valid=False, reference_exact=False, raw_exact=False,
            absolute_reference_expected_score_error=None, errors=['invalid_json'])
    formatted = set(value) == {'move', 'pv', 'candidates', 'evaluation'}
    if not formatted: errors.append('numeric_four_field_schema_required')
    move = value.get('move'); first_legal = isinstance(move, str) and move in legal
    if not first_legal: errors.append('illegal_or_missing_move')
    pv = value.get('pv'); pv_legal = False; complete = False
    if (isinstance(pv, list) and 1 <= len(pv) <= 6 and
            all(isinstance(m, str) for m in pv) and pv[0] == move):
        try:
            continuation_positions(row, dict(pv=pv))
            pv_legal = True
            complete = len(pv) == 6 or adjudicate(row['initial_fen'], [*row['moves'], *pv])['ended']
        except (ValueError, TypeError, KeyError): pass
    if not pv_legal: errors.append('invalid_full_history_PV')
    elif not complete: errors.append('short_nonterminal_PV')
    candidates = value.get('candidates')
    candidates_valid = (isinstance(candidates, list) and len(candidates) == min(3, len(legal)) and
        all(isinstance(m, str) and m in legal for m in candidates) and len(set(candidates)) == len(candidates) and
        bool(candidates) and candidates[0] == move)
    if not candidates_valid: errors.append('invalid_candidates')
    score = value.get('evaluation'); score_valid = False; difference = None
    try:
        if not isinstance(score, dict) or set(score) != {'type', 'value', 'perspective'}:
            raise ValueError('Exact score fields required')
        probability = evaluation_probability(row['fen'], score)
        difference = abs(probability - evaluation_probability(row['fen'], expected['evaluation']))
        score_valid = True
    except (ValueError, KeyError, AttributeError, TypeError, OverflowError): pass
    if not score_valid: errors.append('invalid_evaluation_contract')
    return dict(answer_format_valid=formatted, first_move_legal=first_legal,
        reference_move_match=first_legal and move == expected['move'], pv_history_legal=pv_legal,
        six_ply_or_terminal=complete, candidates_valid=candidates_valid, evaluation_contract_valid=score_valid,
        contract_valid=not errors, reference_exact=not errors and value == expected,
        raw_exact=raw == row['answer'], absolute_reference_expected_score_error=difference, errors=errors)


def prediction_record(row, raw):
    return dict(id=row['id'], record=row, generated=raw, grade=grade(row, raw))


def summary(records):
    if not records or {r['record']['stage'] for r in records} != set(STAGES):
        raise ValueError('Diagnostics require all actual rows from both original validation stages')
    result = dict(examples=len(records), unique_history_contexts=len({r['record']['feature_key'] for r in records}),
        by_stage={}, new_engine_queries=0, raw_answers_repaired=False, reference_is_measured_human_strength=False,
        probability_model='pinned_Pikafish_material_WDL', independent_test_used_for_training_or_selection=False)
    for stage in STAGES:
        rows = [r['grade'] for r in records if r['record']['stage'] == stage]
        keys = ['answer_format_valid', 'first_move_legal', 'reference_move_match', 'contract_valid', 'reference_exact', 'raw_exact']
        if stage == NUMERIC_STAGE: keys += ['pv_history_legal', 'six_ply_or_terminal', 'candidates_valid', 'evaluation_contract_valid']
        values = dict(examples=len(rows), **{key + '_rate': sum(r[key] for r in rows) / len(rows) for key in keys},
            errors=dict(Counter(error for r in rows for error in r['errors'])))
        if stage == NUMERIC_STAGE:
            differences = [r['absolute_reference_expected_score_error'] for r in rows
                           if r['absolute_reference_expected_score_error'] is not None]
            values.update(scored_evaluation_responses=len(differences),
                mean_absolute_reference_expected_score_error_conditional=sum(differences) / len(differences) if differences else None)
        result['by_stage'][stage] = values
    return result


def _selection(data, bound):
    data = Path(data).resolve()
    proof = bound.json(data / 'manifest.json'); counts = proof.get('verification', {})
    if (proof.get('status') != 'complete' or proof.get('kind') != CORPUS_KIND or
            proof.get('config', {}).get('profile') != CORPUS_PROFILE or
            counts.get('original_validation_row_bytes_preserved') is not True or
            counts.get('all_supervised_positions_and_mirrors_isolated') is not True or
            counts.get('test_rows') != 0 or counts.get('final_independent_test_rows_parsed_or_tokenized') != 0):
        raise ValueError('Require the actual completed isolated auxiliary corpus')
    actual = bound.json(data / 'counts.json', proof['outputs'][str(data / 'counts.json')])
    contract = bound.json(data / 'contract.json', proof['outputs'][str(data / 'contract.json')])
    if actual != counts or contract.get('closure') != _closure():
        raise ValueError('Original corpus counts or native reference decoder changed')
    path = data / 'validation.jsonl'; bound.bind(path, proof['outputs'][str(path)])
    observed = Counter(); selected = []; raw_selected = []; ids = set()
    for _, raw, row in _lines(path):
        _basic(row, 'validation', ALL_STAGES)
        if row['id'] in ids: raise ValueError('Original validation IDs must be unique')
        ids.add(row['id']); observed[row['stage']] += 1
        if row['stage'] in STAGES:
            validate_context(row)
            if not grade(row, row['answer'])['reference_exact']:
                raise ValueError('The preserved native reference fails its own diagnostic contract')
            selected.append(row); raw_selected.append(raw)
    if (dict(observed) != counts.get('validation_rows_by_stage') or
            any(observed[stage] < 1 for stage in STAGES) or
            any(len({r['feature_key'] for r in selected if r['stage'] == stage}) != observed[stage] for stage in STAGES)):
        raise ValueError('Diagnostics must cover every distinct original numeric and single-move validation row')
    result = dict(status='complete', all_original_auxiliary_validation_views_selected=True,
        rows_by_stage={s: observed[s] for s in STAGES}, examples=len(selected),
        unique_history_contexts=len({r['feature_key'] for r in selected}), source_validation_rows=sum(observed.values()),
        source_row_bytes_and_order_preserved=True, train_or_final_test_rows_parsed=False, new_engine_or_model_queries=0)
    bound.unchanged()
    return dict(data=str(data), corpus_manifest_sha256=digest(data / 'manifest.json')), result, selected, b''.join(raw_selected)


def prepare(data, output):
    root = Path(output).resolve()
    if root.exists(): raise FileExistsError('Preserve previous diagnostic inputs; use a fresh directory')
    bound = BoundInputs(); logical, result, _, raw = _selection(data, bound)
    root.mkdir(); (root / 'validation.jsonl').write_bytes(raw)
    atomic_json(root / 'counts.json', result)
    completed = manifest(INPUT_KIND, logical,
        outputs=[root / 'validation.jsonl', root / 'counts.json'], verification=result)
    completed['inputs'] = bound.artifacts; atomic_json(root / 'manifest.json', completed)
    return result


def checked_inputs(root, bound):
    root = Path(root); proof = bound.json(root / 'manifest.json')
    if proof.get('kind') != INPUT_KIND or proof.get('status') != 'complete':
        raise ValueError('Require completed original validation diagnostic inputs')
    original = BoundInputs(); logical, result, selected, raw = _selection(proof['config']['data'], original)
    if proof.get('config') != logical or proof.get('verification') != result or proof.get('inputs') != original.artifacts:
        raise ValueError('Prepared diagnostics must match the complete original source validation')
    for section in ['inputs', 'outputs']:
        for path, identity in proof[section].items(): bound.bind(path, identity)
    if (bound.json(root / 'counts.json') != result or (root / 'validation.jsonl').read_bytes() != raw or
            any(str(root / name) not in proof['outputs'] for name in ['validation.jsonl', 'counts.json'])):
        raise ValueError('Prepared diagnostics changed original row bytes, order or complete coverage')
    return selected


def run_evaluation(checkpoint, completion, features, inputs, output, *, expert_weights,
                   batch_size=8, move_budget=64, numeric_budget=768, memory='normal', resume=False):
    from .auxiliary_policy_preflight import inherited_cache
    if any(type(n) is not int or n < 1 for n in [batch_size, move_budget, numeric_budget]) or memory not in {'normal', 'zero', 'shuffled'}:
        raise ValueError('Use positive diagnostic budgets and a declared memory condition')
    if memory == 'shuffled' and batch_size < 2: raise ValueError('Shuffled batches need two distinct histories')
    checkpoint, completion, features, inputs, root = map(Path, [checkpoint, completion, features, inputs, output])
    bound = BoundInputs(); rows = checked_inputs(inputs, bound)
    saved, parent = completed_parent(checkpoint, completion, bound)
    for path in _decoder_assets(Path(saved['model_path'])).values(): bound.bind(path)
    bound.bind(expert_weights)
    if features.resolve() != Path(saved['feature_path']).resolve(): cache, _ = inherited_cache(features, saved, bound)
    else:
        for path in feature_paths(features): bound.bind(path)
        cache = FeatureStore(features, verified_hashes={p: v['sha256'] for p, v in bound.artifacts.items()})
    keys = set(cache.keys)
    if (cache.depths != saved['expert_feature_depths'] or cache.proof['verification']['expert_sha256'] != bound.artifacts[str(expert_weights)]['sha256'] or
            any(row['feature_key'] not in keys for row in rows)):
        raise ValueError('All fixed validation histories must use the declared trained expert and actual cache')
    contract = dict(checkpoint=str(checkpoint), completion=str(completion), features=str(features), inputs=str(inputs),
        expert_weights=str(expert_weights), batch_size=batch_size, move_budget=move_budget, numeric_budget=numeric_budget,
        memory=memory, output=str(root), code=code_identity(), input_bindings=dict(bound.artifacts))
    ordered = [row for stage in STAGES for row in rows if row['stage'] == stage]
    def checked_records(records):
        if len(records) > len(ordered) or any(r != prediction_record(row, r['generated']) for r, row in zip(records, ordered)):
            raise ValueError('Preserve changed, missing or repaired diagnostic raw records')
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only the exact frozen diagnostic contract')
        if (root / 'manifest.json').exists():
            proof = json.loads((root / 'manifest.json').read_text()); checked_artifacts(proof)
            records = load_jsonl(root / 'predictions.jsonl')
            checked_records(records)
            if (len(records) != len(ordered) or records != load_jsonl(root / 'predictions.partial.jsonl') or
                    proof.get('kind') != KIND or proof.get('config') != contract or
                    any(proof['verification'].get(k) != v for k, v in summary(records).items())):
                raise ValueError('Completed diagnostics differ from their preserved raw answers')
            return proof['verification']
    else: root.mkdir(parents=True); atomic_json(root / 'contract.json', contract)
    partial = root / 'predictions.partial.jsonl'; records = load_jsonl(partial) if partial.exists() else []
    checked_records(records)
    stage = 'student_load'; start = time.monotonic()
    try:
        atomic_json(root / 'state.json', dict(status=stage, completed_rows=len(records)))
        predictor = None
        if len(records) < len(ordered):
            from .inference import Predictor
            predictor = Predictor(str(checkpoint), weights=str(expert_weights), feature_cache=str(features))
            if predictor.config != saved: raise ValueError('The loaded diagnostic model differs from its declared completed state')
        for task in STAGES:
            remaining = [row for row in ordered[len(records):] if row['stage'] == task]
            offset = 0
            while offset < len(remaining):
                size = batch_size
                if memory == 'shuffled' and len(remaining) - offset == batch_size + 1: size += 1
                batch = remaining[offset:offset + size]
                if memory == 'shuffled' and len(batch) == 1: raise ValueError('No singleton shuffled validation batch is allowed')
                budget = numeric_budget if task == NUMERIC_STAGE else move_budget
                stage = 'raw_' + task
                answers = predictor.generate_batch([prompt_record(row) for row in batch], budget, memory)
                if len(answers) != len(batch): raise ValueError('Every requested raw answer, including errors, must be retained')
                fresh = [prediction_record(row, raw) for row, raw in zip(batch, answers, strict=True)]
                with partial.open('a') as handle:
                    for record in fresh: handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                records.extend(fresh); offset += len(batch); bound.unchanged()
                if code_identity() != contract['code']: raise ValueError('Diagnostic execution source changed')
                atomic_json(root / 'state.json', dict(status=stage, completed_rows=len(records), total_rows=len(ordered)))
        metrics = dict(summary(records), seconds=time.monotonic() - start,
            trained_auxiliary_policy_rounds=parent['verification'].get('trained_auxiliary_policy_rounds', 0),
            trained_search_distillation_rounds=parent['verification'].get('trained_search_distillation_rounds', 0),
            generation_end_reason_exposed=False, strong_chess_or_reliable_teaching_proven=False)
        write_jsonl(root / 'predictions.jsonl', records); atomic_json(root / 'metrics.json', metrics)
        completed = manifest(KIND, contract, outputs=[partial, root / 'predictions.jsonl', root / 'metrics.json'], verification=metrics)
        completed['inputs'] = bound.artifacts; atomic_json(root / 'manifest.json', completed)
        atomic_json(root / 'state.json', dict(status='complete', completed_rows=len(records)))
        return metrics
    except Exception as error:
        atomic_json(root / 'state.json', dict(status='failed_preserve_raw_outputs', stage=stage, error_type=type(error).__name__))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='action', required=True)
    prepare_parser = commands.add_parser('prepare'); prepare_parser.add_argument('--data', required=True)
    prepare_parser.add_argument('--output', required=True)
    evaluate = commands.add_parser('evaluate')
    for key in ['checkpoint', 'completion', 'features', 'inputs', 'output', 'expert-weights']: evaluate.add_argument('--' + key, required=True)
    for key, default in [('batch-size', 8), ('move-budget', 64), ('numeric-budget', 768)]: evaluate.add_argument('--' + key, type=int, default=default)
    evaluate.add_argument('--memory', choices=['normal', 'zero', 'shuffled'], default='normal'); evaluate.add_argument('--resume', action='store_true')
    args = vars(parser.parse_args()); action = args.pop('action')
    with input_session_from_environment(): print(json.dumps(prepare(**args) if action == 'prepare' else run_evaluation(**args)), flush=True)


if __name__ == '__main__': main()
