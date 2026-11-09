"""Admission of separately recovered CPU scores; never relabel the failed run."""
import json
from pathlib import Path

from xqgeneral.evidence import code_identity, digest, load_jsonl
from xqgeneral.evaluate_explanations import explanation_summary
from xqgeneral.explanations import EXPLANATION_QUESTION
from xqgeneral.sft_preflight import checked_artifacts


def checked_recovered_raw(root, memory, rows, args, core, *, expected_code=None):
    root = Path(root)
    proof = json.loads((root / 'manifest.json').read_text())
    if (proof.get('status') != 'complete' or proof.get('kind') != 'recovered_independent_explanation_evaluation' or
            args.get('output') != str(root) or
            proof['code'] != (code_identity() if expected_code is None else expected_code) or
            proof['config'].get('generation_config') != {k:v for k,v in args.items() if k != 'output'}):
        raise ValueError('Recovered scoring must bind the exact original generation and new scoring source')
    core.unchanged()
    checked_artifacts(proof)
    receipt_path = Path(proof['config']['receipt'])
    receipt = json.loads(receipt_path.read_text())
    if (receipt['kind'] != 'preserved_raw_explanation_generation' or
            receipt['status'] != 'complete_raw_answers_only' or
            receipt['generation_config'] != proof['config']['generation_config'] or
            proof['verification']['saved_raw_generation_receipt_sha256'] != digest(receipt_path) or
            digest(root/'raw-predictions.jsonl') != receipt['raw_predictions']['sha256']):
        raise ValueError('Recovered scores must preserve the complete original raw generation receipt')
    if (proof['inputs'].get(str(receipt_path)) !=
            {'sha256':digest(receipt_path),'bytes':receipt_path.stat().st_size} or
            proof['inputs'].get(receipt['raw_predictions']['path']) !=
            {k:receipt['raw_predictions'][k] for k in ('sha256','bytes')}):
        raise ValueError('Recovered scores must bind both the receipt and its actual raw origin')
    for path in [str(Path(args['data'])/'validation.jsonl'),args['executable'],args['weights']]:
        if path not in core.expected or proof['inputs'].get(path) != core.expected[path]:
            raise ValueError('Recovered scoring validation or engine identity differs')
    raw, judged = load_jsonl(root/'raw-predictions.jsonl'), load_jsonl(root/'judged-predictions.jsonl')
    expected = {r['id']:dict(r,question=EXPLANATION_QUESTION) for r in rows}
    by_id = {r['id']:r for r in raw}
    if (len(expected) != len(rows) or len(raw) != len(rows) or len(by_id) != len(rows) or
            set(by_id) != set(expected) or any(r['record'] != expected[r['id']] for r in raw) or
            len(judged) != len(rows) or len({r['id'] for r in judged}) != len(rows) or
            any(r['id'] not in by_id or {k:r[k] for k in ('id','record','raw')} != by_id[r['id']]
                for r in judged)):
        raise ValueError('Recovered scoring must retain all exact protected questions and unmodified answers')
    metrics = json.loads((root/'metrics.json').read_text())
    if (metrics != proof['verification'] or
            any(metrics.get(k) != v for k,v in explanation_summary(judged).items()) or
            metrics.get('memory') != memory or metrics.get('split') != 'validation' or
            metrics.get('judge_nodes_per_position') != args['nodes'] or
            metrics.get('new_model_or_teacher_queries') != 0 or
            metrics.get('original_generation_code') != receipt['generation_code'] or
            metrics.get('engine_sha256') != core.expected[args['executable']]['sha256'] or
            metrics.get('engine_weights_sha256') != core.expected[args['weights']]['sha256']):
        raise ValueError('Recovered aggregates must match all preserved rows, original budgets and scoring coverage')
    core.unchanged()
    return metrics, judged
