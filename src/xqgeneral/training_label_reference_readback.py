"""Reparse retained engine requests and independently recompute label-reference losses."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .calibration import candidate_probability, evaluation_probability
from .evidence import digest
from .explanations import parse_explanation, validate_explanation
from .oracle import parse_analysis
from .reference_oracle import RECOVERY_POLICY
from .reviewed_explanations import checked_label
from .rules import adjudicate, legal_moves, play
from .training_label_reference import (KIND, check_bindings, load_training, read,
                                        row_binding, validate_plan, write_once)


FLAGS = ['all_original_label_records_and_full_native_histories_verified',
         'all_stored_engine_responses_reparsed', 'all_complete_pv_loss_aggregates_recomputed',
         'original_unscored_and_same_best_additional_responses_verified']
COUNT_KEYS = ['parse_valid', 'structured_contract_valid', 'first_move_legal',
              'first_move_no_mistake', 'pv_legal', 'full_pv_no_mistake']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_row(value, record, index, binding):
    """Verify histories, ordered requests, original responses and every stored loss."""
    require(value.get('row_index') == index and value.get('id') == record['id'] and
            value.get('record') == record and value.get('raw') == record['answer'] and
            value.get('audit_binding') == binding, 'Reference row identity or original label changed')
    checked_label(record)
    analysis = parse_explanation(value['raw'])
    verdict = value['judgment']
    contract = validate_explanation(record['fen'], analysis, require_branches=True)
    require(verdict.get('parse_valid') is True and verdict.get('contract_valid') is contract['valid'] and
            verdict.get('errors') == contract['errors'], 'The retained structured-label verdict differs')
    requests, recoveries = value['engine_requests'], value['oracle_exact_score_recoveries']
    require(isinstance(requests, list) and requests and isinstance(recoveries, list),
            'Every actual engine request and additional-score request must be retained')
    cursor = recovery_cursor = 0

    def request(fen, moves, searchmoves):
        nonlocal cursor
        require(cursor < len(requests), 'A required retained engine request is absent')
        retained = requests[cursor]
        cursor += 1
        require(all(retained.get(key) == expected for key, expected in dict(
            fen=fen, nodes=binding['nodes_per_position'], initial_fen=record['initial_fen'],
            moves=moves, searchmoves=searchmoves).items()),
            'Engine request differs from the original full history, move restriction or node budget')
        return retained

    def engine(fen, moves, searchmoves=None):
        nonlocal recovery_cursor
        original = request(fen, moves, searchmoves)
        try:
            return parse_analysis(fen, original['raw_output'], binding['nodes_per_position'])
        except RuntimeError as error:
            if str(error) != 'Engine completed without a scored best-move candidate' or searchmoves:
                raise
        require(recovery_cursor < len(recoveries), 'The original unscored request has no explicit recovery')
        recovery = recoveries[recovery_cursor]
        recovery_cursor += 1
        selected = original['raw_output'][-1].split()[1]
        additional = request(fen, moves, [selected])
        exact = parse_analysis(fen, additional['raw_output'], binding['nodes_per_position'])
        require(exact['best_move'] == selected and all(recovery.get(key) == expected for key, expected in dict(
            row_index=index, policy=RECOVERY_POLICY, original_request=original,
            additional_request=additional, original_selected_move=selected, selected_move_unchanged=True,
            strict_result=exact, additional_engine_requests=1,
            teacher_or_student_prediction_rewritten=False).items()),
            'The extra request changed the original best move or its retained strict score')
        return exact

    def loss(fen, move, oracle, moves, restricted):
        best = next(candidate for candidate in oracle['candidates'] if candidate['move'] == oracle['best_move'])
        chosen = next((candidate for candidate in oracle['candidates'] if candidate['move'] == move), None)
        if chosen is None:
            exact = engine(fen, moves, [move])
            require(exact == restricted and exact['best_move'] == move,
                    'A restricted score differs from the same original proposed move')
            chosen = next(candidate for candidate in exact['candidates'] if candidate['move'] == move)
        else:
            require(restricted is None, 'An unused restricted request was added')
        return max(0, candidate_probability(fen, best) - candidate_probability(fen, chosen))

    position, moves = record['fen'], list(record['moves'])
    require(not adjudicate(record['initial_fen'], moves)['ended'], 'An explanation root is terminal')
    require(verdict.get('first_move_legal') is True and analysis['move'] in legal_moves(position),
            'The original recommended move is not legal')
    root = engine(position, moves)
    require(root == verdict['first_move_oracle'], 'The first-move oracle differs from its retained output')
    first = loss(position, analysis['move'], root, moves, verdict.get('first_move_restricted_oracle'))
    require(verdict.get('first_move_expected_score_loss') == first and
            verdict.get('first_move_no_mistake') is (first < .10), 'The first-move loss or threshold differs')
    best = next(candidate for candidate in root['candidates'] if candidate['move'] == root['best_move'])
    try:
        evaluation_error = abs(evaluation_probability(position, analysis.get('evaluation')) -
                               candidate_probability(position, best))
    except (ValueError, TypeError, AttributeError):
        require('evaluation_probability_error' not in verdict, 'An invalid evaluation was silently scored')
    else:
        require(verdict.get('evaluation_probability_error') == evaluation_error,
                'The retained evaluation probability error differs')
    pv = analysis['pv']
    require(isinstance(pv, list) and 0 < len(pv) <= 32 and
            len(verdict['engine_judgments']) == len(pv), 'The complete main variation was not scored')
    losses = []
    for offset, move in enumerate(pv):
        require(not adjudicate(record['initial_fen'], moves)['ended'] and move in legal_moves(position),
                'The main variation is illegal or continues after the native game end')
        stored = verdict['engine_judgments'][offset]
        require(stored.get('fen') == position and stored.get('move') == move,
                'A retained variation judgment differs from the original variation')
        oracle = root if offset == 0 else engine(position, moves)
        require(oracle == stored['oracle'], 'A variation score differs from its retained engine output')
        losses.append(loss(position, move, oracle, moves, stored['restricted']))
        position = play(position, move)
        moves.append(move)
    require(cursor == len(requests) and recovery_cursor == len(recoveries),
            'Unused engine requests or extra-score records are not part of the original judgment')
    require(verdict.get('pv_legal') is True and verdict['pv_move_losses'] == losses,
            'The retained main-variation legality or step losses differ')
    counts = dict(parse_valid=1, structured_contract_valid=int(contract['valid']), first_move_legal=1,
                  first_move_no_mistake=int(first < .10), pv_legal=1,
                  full_pv_no_mistake=int(all(value < .10 for value in losses)))
    return dict(counts=counts, first_loss=first, pv_losses=losses,
                additional_requests=recovery_cursor, actual_engine_requests=cursor)


def producer(reference, plan):
    config = plan['config']
    destination = reference / config['scope']
    proof = read(destination / 'manifest.json')
    metrics = read(destination / 'metrics.json')
    require(proof.get('kind') == KIND and proof.get('status') == 'complete' and
            proof.get('config') == dict(config, plan_sha256=digest(reference / 'plan.json')) and
            proof.get('inputs') == plan['inputs'] and proof.get('metrics') == metrics and
            proof.get('loss_cutoff') == 'expected_score_loss_strictly_less_than_0.10',
            'The completed producer manifest differs from its pinned plan or metrics')
    check_bindings(proof['outputs'])
    expected = {str(destination / 'rows' / f'{index:06}.json') for index in config['indices']}
    expected.add(str(destination / 'metrics.json'))
    require(set(proof['outputs']) == expected, 'The producer output coverage differs')
    exit_proof = read(destination / 'exit.json')
    require(exit_proof.get('returncode') == 0 and
            exit_proof.get('manifest_sha256') == digest(destination / 'manifest.json'),
            'The producer did not finish with this exact manifest')
    return proof, metrics


def completed_readback(reference, plan=None):
    """Rebind an already completed reader receipt; do not claim a new math pass."""
    reference = Path(reference).resolve()
    actual_plan = validate_plan(reference)
    require(plan is None or plan == actual_plan, 'The completed reader was given a different pinned plan')
    plan = actual_plan
    proof, metrics = producer(reference, plan)
    destination = reference / plan['config']['scope']
    result = read(destination / 'independent-readback.json')
    require(result.get('status') == 'complete_training_label_reference_independent_readback' and
            result.get('scope') == plan['config']['scope'] and
            result.get('examples') == len(plan['config']['indices']) and
            result.get('producer_manifest_sha256') == digest(destination / 'manifest.json') and
            result.get('plan_sha256') == digest(reference / 'plan.json') and
            result.get('script_sha256') == digest(reference / 'readback.py') and
            result.get('reader_module_sha256') == digest(Path(__file__)) and
            all(result.get(key) is True for key in FLAGS) and
            result.get('labels_or_predictions_rewritten') is False and
            result.get('validation_or_independent_test_answers_read') is False and
            result.get('actual_engine_calls_in_this_reader') == 0 and
            set(result.get('actual_retained_metric_counts', {})) == set(COUNT_KEYS),
            'The completed independent reference receipt differs')
    recomputed = result.get('recomputed_metrics', {})
    required = {key + '_rate' for key in COUNT_KEYS} | {
        'examples', 'full_pv_no_mistake_count', 'judged_legal_pv_plies',
        'legal_scored_first_moves', 'mean_first_move_expected_score_loss_conditional',
        'pv_no_mistake_fraction_conditional', 'errors', 'by_color_derivation'}
    require(required <= set(recomputed) and all(metrics.get(key) == value for key, value in recomputed.items()),
            'The independent recomputed metrics differ from the producer')
    return result


def readback(reference):
    reference = Path(reference).resolve()
    plan = validate_plan(reference)
    destination = reference / plan['config']['scope']
    if (destination / 'independent-readback.json').exists():
        completed_readback(reference, plan)
        return dict(status='completed_reader_receipt_rebound_without_new_math_pass',
                    examples=len(plan['config']['indices']))
    proof, metrics = producer(reference, plan)
    rows = load_training(plan['config']['data'], plan['config']['artifact_root'])
    require(len(rows) == plan['config']['source_training_rows'], 'The original training count changed')
    counts = Counter({key: 0 for key in COUNT_KEYS})
    first_losses, pv_losses = [], []
    groups = {name: Counter(examples=0, first_move_no_mistake=0, full_pv_no_mistake=0)
              for name in ['with_augmentation_parent', 'without_augmentation_parent']}
    additional = requests = 0
    binding = row_binding(reference, plan)
    for index in plan['config']['indices']:
        value = read(destination / 'rows' / f'{index:06}.json')
        result = verify_row(value, rows[index], index, binding)
        counts.update(result['counts'])
        first_losses.append(result['first_loss'])
        pv_losses.extend(result['pv_losses'])
        additional += result['additional_requests']
        requests += result['actual_engine_requests']
        group = groups['with_augmentation_parent' if rows[index].get('augmentation_parent')
                       else 'without_augmentation_parent']
        group.update(examples=1, first_move_no_mistake=result['counts']['first_move_no_mistake'],
                     full_pv_no_mistake=result['counts']['full_pv_no_mistake'])
    count = len(plan['config']['indices'])
    recomputed = {key + '_rate': value / count for key, value in counts.items()}
    recomputed.update(examples=count, full_pv_no_mistake_count=counts['full_pv_no_mistake'],
        legal_scored_first_moves=len(first_losses),
        mean_first_move_expected_score_loss_conditional=sum(first_losses) / len(first_losses),
        judged_legal_pv_plies=len(pv_losses),
        pv_no_mistake_fraction_conditional=sum(value < .10 for value in pv_losses) / len(pv_losses),
        errors={}, by_color_derivation={name: dict(value) for name, value in groups.items()})
    require(all(metrics.get(key) == value for key, value in recomputed.items()) and
            metrics['additional_best_score_requests'] == additional and
            metrics['nodes_per_engine_request'] == plan['config']['nodes'] and
            metrics['exact_score_recovery_policy'] == RECOVERY_POLICY,
            'Independent native losses or summary metrics do not reproduce the producer')
    for key in ['raw_generation', 'student_model_loaded_or_evaluated', 'gpu_used',
                'teacher_labels_changed', 'independent_test_answers_loaded', 'validation_answers_loaded']:
        require(metrics.get(key) is False, 'The producer scope differs from a train-only label reference')
    require(metrics.get('split') == 'train' and metrics.get('oracle_repairs') == 0,
            'Reference labels were repaired or taken from another split')
    validate_plan(reference)
    check_bindings(proof['outputs'])
    result = dict(status='complete_training_label_reference_independent_readback',
        observed_at_utc=datetime.now(timezone.utc).isoformat(), scope=plan['config']['scope'], examples=count,
        actual_retained_metric_counts=dict(counts), recomputed_metrics=recomputed,
        **{key: True for key in FLAGS}, retained_engine_requests_verified=requests,
        stored_engine_response_objects_reparsed=requests,
        explicitly_verified_additional_engine_requests=additional,
        all_ordered_engine_requests_bound_to_original_full_histories=True,
        labels_or_predictions_rewritten=False, validation_or_independent_test_answers_read=False,
        actual_engine_calls_in_this_reader=0, engine_search_or_model_inference_started=False,
        producer_manifest_sha256=digest(destination / 'manifest.json'),
        plan_sha256=digest(reference / 'plan.json'), script_sha256=digest(reference / 'readback.py'),
        reader_module_sha256=digest(Path(__file__)),
        human_rating=False, student_strength_or_coaching_benefit_proven=False)
    write_once(destination / 'independent-readback.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    args = parser.parse_args()
    result = readback(args.reference)
    print(f"{result['status']}: {result['examples']} training labels; zero engine calls")


if __name__ == '__main__':
    main()
