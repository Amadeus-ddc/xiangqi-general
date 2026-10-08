"""Best raw-validation checkpoint selection, with explicit local stopping rules."""
from pathlib import Path
import math

from .curriculum_data import STAGES
from .evidence import load_jsonl


COURSE_KIND = 'validation_selected_foundation_course'
CURRICULUM_KIND = 'four_course_validation_selected_curriculum'
HANDOFF_KIND = 'validation_selected_foundation_handoff'


def validate_policy(policy):
    if (set(policy) != {'metric', 'tie_breaker', 'patience'} or
            policy['metric'] != 'balanced_raw_qa_accuracy' or
            policy['tie_breaker'] != 'minimum_task_accuracy_then_earliest' or
            type(policy['patience']) is not int or policy['patience'] < 1):
        raise ValueError('Declare the raw-validation metric, tie breaker and positive patience')


def selection_decision(history, policy, budget):
    """Select by held-out raw answers, never NLL or the most recent training update."""
    validate_policy(policy)
    if not history:
        raise ValueError('Checkpoint selection requires actual validation candidates')
    steps = [row['step'] for row in history]
    expected = list(range(budget['qa_every'], steps[-1] + 1, budget['qa_every']))
    if steps[-1] == budget['steps'] and steps[-1] % budget['qa_every']:
        expected.append(steps[-1])
    if (any(type(s) is not int or s <= 0 for s in steps) or steps != expected or
            steps[-1] > budget['steps']):
        raise ValueError('Validation history must cover every scheduled check in update order')
    for row in history:
        gate = row['gate']
        if (gate.get('validation_only') is not True or gate.get('raw_generation') is not True or
                gate.get('oracle_used') is not False or not gate['by_task']):
            raise ValueError('Selection requires normal-memory raw validation without an oracle')
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
               for value in [gate['accuracy'], *[task['accuracy'] for task in gate['by_task'].values()]]):
            raise ValueError('Raw validation accuracies must be finite fractions')
    eligible = [row for row in history if row['step'] >= budget['minimum_steps']]
    if not eligible:
        return {'stop': False, 'selected_step': None, 'steps_completed': steps[-1],
                'checks_without_improvement': 0, 'stopping_reason': None}
    def score(row):
        return row['gate']['accuracy'], min(t['accuracy'] for t in row['gate']['by_task'].values()), -row['step']
    best = max(eligible, key=score)
    stale = sum(row['step'] > best['step'] for row in eligible)
    reason = ('reference_targets_met' if best['gate']['passed'] else
              'validation_plateau' if stale >= policy['patience'] else
              'maximum_budget' if steps[-1] == budget['steps'] else None)
    return {'stop': reason is not None, 'selected_step': best['step'], 'steps_completed': steps[-1],
            'checks_without_improvement': stale, 'stopping_reason': reason,
            'selected_accuracy': best['gate']['accuracy'],
            'selected_minimum_task_accuracy': score(best)[1]}


def course_contract(recipe, index, root, parent, artifacts, execution_code):
    """A first-course import keeps its original source/config; later courses inherit its bytes."""
    from .foundation_preflight import clean_recipe
    from .gated_curriculum import training_config
    stage = STAGES[index]
    imported = recipe.get('initial_course_import') if index == 0 else None
    if not imported:
        return (training_config(recipe, index, root, parent), root / stage / 'candidates', execution_code)
    if (set(imported) != {'recipe', 'through_step', 'source_manifest'} or
            type(imported['through_step']) is not int or imported['through_step'] < 1):
        raise ValueError('Import only a declared clean first course and its preserved execution source')
    original = artifacts.json(imported['recipe'])
    clean_recipe(original)
    if original.get('validation_selection') or original.get('initial_course_import'):
        raise ValueError('Initial import must come directly from the original clean foundation recipe')
    original_root = Path(original['output'])
    if original_root.resolve() == root.resolve():
        raise ValueError('Changed selection policy requires a fresh curriculum output')
    expected = training_config(original, 0, original_root, None)
    if training_config(recipe, 0, original_root, None) != expected or recipe['raw_qa_gates'] != original['raw_qa_gates']:
        raise ValueError('Imported course training, data and raw validation contracts must stay unchanged')
    source_path = Path(imported['source_manifest'])
    source = artifacts.json(source_path)
    code = {'revision': source['revision'],
            'source_sha256': {f'src/xqgeneral/{name}': sha for name, sha in source['source_sha256'].items()}}
    for name, sha in source['source_sha256'].items():
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Imported execution source contains an invalid file identity')
        path = source_path.parent / 'source/xqgeneral' / name
        artifacts.add(path, {'sha256': sha, 'bytes': path.stat().st_size})
    original_config = original_root / stage / 'training.config.json'
    if artifacts.json(original_config) != expected:
        raise ValueError('Original course configuration changed before import')
    return expected, original_root / stage / 'candidates', code


def read_history(candidate_root, through_step, expected_config, training_code, stages, recipe, artifacts):
    """Bind every compared checkpoint to its frozen snapshot and unmodified raw answers."""
    import torch
    from .gated_curriculum import raw_gate_from_artifacts
    interval = recipe['course_budgets'][len(stages) - 1]['qa_every']
    steps = list(range(interval, through_step + 1, interval))
    if not steps or steps[-1] != through_step:
        if through_step != recipe['course_budgets'][len(stages) - 1]['steps']:
            raise ValueError('Only the maximum budget may end between scheduled raw checks')
        steps.append(through_step)
    validation = load_jsonl(Path(recipe['data_path']) / 'validation.jsonl')
    history = []
    for step in steps:
        candidate = Path(candidate_root) / f'step-{step}'
        checkpoint = candidate / 'adapter.pt'
        snapshot = artifacts.json(candidate / 'snapshot.json')
        if (snapshot['actual_step'] != step or snapshot['training_config'] != expected_config or
                snapshot['training_code'] != training_code or
                snapshot.get('atomic_source_inode_pinned') is not True or
                snapshot.get('optimizer_state_in_adapter') is not False):
            raise ValueError('Validation candidate differs from the declared frozen training snapshot')
        artifacts.add(checkpoint, {'sha256': snapshot['adapter_sha256'], 'bytes': checkpoint.stat().st_size})
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        inputs = [recipe['feature_path'], *[str(Path(recipe['data_path']) / f'{s}.jsonl') for s in ['train', 'validation']]]
        if expected_config.get('init_from'):
            inputs.append(expected_config['init_from'])
        if (saved['config'] != expected_config or saved['selected_step'] != step or
                saved['code'] != training_code or saved['input_hashes'] != snapshot['training_input_hashes'] or
                set(saved['input_hashes']) != set(inputs)):
            raise ValueError('Actual candidate weights differ from their step, source or input lineage')
        del saved
        for name, sha in snapshot['training_input_hashes'].items():
            artifacts.add(name, {'sha256': sha, 'bytes': Path(name).stat().st_size})
        qa = artifacts.proof(candidate / 'qa/manifest.json', 'balanced_board_qa')
        if qa['code'] != training_code:
            raise ValueError('Raw evaluation must use the candidate training execution source')
        required_inputs = [checkpoint, Path(recipe['data_path']) / 'validation.jsonl', Path(recipe['feature_path'])]
        required_outputs = [candidate / 'qa/predictions.jsonl', candidate / 'qa/metrics.json']
        if (any(str(p) not in qa['inputs'] for p in required_inputs) or
                any(str(p) not in qa['outputs'] for p in required_outputs)):
            raise ValueError('Raw validation lacks required input or prediction bindings')
        metrics = artifacts.json(candidate / 'qa/metrics.json')
        if metrics != qa['verification'] or metrics.get('split') != 'validation':
            raise ValueError('Actual raw validation metrics differ from their completed proof')
        if (metrics.get('oracle_repairs', 0) or metrics.get('legality_is_imposed_by_decoding', False) or
                metrics.get('rule_legal_constraints', False)):
            raise ValueError('Selection requires unmodified raw model answers')
        gate = raw_gate_from_artifacts(qa, metrics, load_jsonl(candidate / 'qa/predictions.jsonl'),
            checkpoint, stages, recipe['raw_qa_gates'], recipe['data_path'], recipe['feature_path'], validation)
        if artifacts.json(candidate / 'gate.json') != gate:
            raise ValueError('Stored candidate gate differs from the actual raw validation answers')
        history.append({'step': step, 'checkpoint': str(checkpoint),
                        'checkpoint_sha256': snapshot['adapter_sha256'], 'gate': gate})
    return history


def validate_course_selection(result):
    selection = result['selection']
    decision = selection_decision(selection['history'], selection['policy'], selection['budget'])
    chosen = next(row for row in selection['history'] if row['step'] == decision['selected_step']) if decision['stop'] else None
    if (not decision['stop'] or selection['decision'] != decision or
            result['actual_steps'] != decision['steps_completed'] or result['selected_step'] != decision['selected_step'] or
            result['selected_checkpoint'] != chosen['checkpoint'] or
            result['selected_checkpoint_sha256'] != chosen['checkpoint_sha256'] or result['gate'] != chosen['gate'] or
            result.get('reference_targets_passed') != chosen['gate']['passed']):
        raise ValueError('Course completion must select the actual best raw-validation checkpoint')
    return decision
