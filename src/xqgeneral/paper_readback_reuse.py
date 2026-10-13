"""Reuse a completed native readback only for its exact, freshly hashed corpus."""
from copy import deepcopy
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path

from .course_tasks import PAPER_PROFILE
from .course_sampling import make_sampler
from .course_position_sampling import position_budgets, position_verification
from .evidence import digest, file_signature
from .paper_curriculum import DATA_KIND

KIND = 'paper_curriculum_complete_source_readback'


def _document(path):
    before = file_signature(path)
    raw = Path(path).read_bytes()
    binding = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    if file_signature(path) != before:
        raise ValueError('Completed readback document changed during verification')
    return json.loads(raw), binding, before


def _bindings(values):
    normalized = {}
    for name, value in values.items():
        path = Path(name).resolve()
        if (not isinstance(value, dict) or set(value) != {'sha256', 'bytes'} or
                type(value['bytes']) is not int or value['bytes'] < 0 or
                not isinstance(value['sha256'], str) or len(value['sha256']) != 64):
            raise ValueError('Completed readback artifact binding is invalid')
        if path in normalized and normalized[path] != value:
            raise ValueError('Completed readback has conflicting artifact bindings')
        normalized[path] = value
    return normalized


@dataclass(frozen=True)
class VerifiedPaperReadback:
    data: Path
    corpus: dict
    verification: dict
    manifest_sha256: str
    inputs: tuple
    signatures: dict

    def check_unchanged(self):
        try:
            changed = any(file_signature(path) != signature for path, signature in self.signatures.items())
        except OSError as error:
            raise ValueError('Reused native readback artifact changed or disappeared') from error
        if changed:
            raise ValueError('Reused native readback artifact changed after verification')

    def for_corpus(self, data, proof):
        if Path(data).resolve() != self.data or proof != self.corpus:
            raise ValueError('Verified readback belongs to another corpus identity')
        self.check_unchanged()
        return deepcopy(self.verification)


def verify_completed_readback(data, completed_manifest):
    """Require the same producer, every input/output binding and full native scope.

    This does not regenerate answers. It verifies a prior complete regeneration,
    rejects even a rehashed replacement corpus, and carries its entire source
    artifact graph forward into the caller's new manifest.
    """
    data, completed_manifest = Path(data).resolve(), Path(completed_manifest).resolve()
    corpus_path = data / 'manifest.json'
    corpus, corpus_binding, corpus_signature = _document(corpus_path)
    proof, proof_binding, proof_signature = _document(completed_manifest)
    if (corpus.get('schema_version') != 1 or corpus.get('status') != 'complete' or
            corpus.get('kind') != DATA_KIND or corpus.get('config', {}).get('task_profile') != PAPER_PROFILE or
            Path(corpus['config']['output']).resolve() != data):
        raise ValueError('Completed readback requires the declared paper corpus identity')
    source = corpus['verification']
    if (source.get('all_supplied_roots_and_futures_natively_verified') is not True or
            source.get('prior_corpus_isolation_verified') is not True):
        raise ValueError('Completed readback corpus lacks complete native verification or isolation')
    # This receipt proves every available question, including reported capacity
    # shortfalls. The full training preflight separately requires every task.
    if (proof.get('schema_version') != 1 or proof.get('status') != 'complete' or proof.get('kind') != KIND or
            Path(proof.get('config', {}).get('data', '')).resolve() != data or
            Path(proof.get('config', {}).get('output', '')).resolve() != completed_manifest.parent):
        raise ValueError('Completed native readback proof has another scope or corpus identity')
    code = corpus.get('code', {})
    if not code.get('source_sha256') or proof.get('code') != code:
        raise ValueError('Completed readback producer code identity differs from its corpus')
    counts = source['counts']
    if (set(counts) != {'train', 'validation', 'test'} or
            any(type(n) is not int or n <= 0 for n in counts.values())):
        raise ValueError('Completed readback corpus counts are invalid')
    expected = {'status': 'complete', 'task_profile': PAPER_PROFILE,
                'questions_regenerated_and_compared': sum(counts.values()), 'counts': counts,
                'split_verification': source['split_verification'], 'all_source_and_question_bytes_checked': True,
                'student_model_loaded': False, 'student_training_executed': False, 'student_strength_measured': False}
    verified = proof.get('verification', {})
    if any(verified.get(k) != v for k, v in expected.items()):
        raise ValueError('Completed readback verification is partial or differs from corpus counts')
    if (source['split_verification'].get('game_overlap') != 0 or
            source['split_verification'].get('root_and_future_overlap') != 0):
        raise ValueError('Completed readback corpus verification contains heldout overlap')
    sampler = make_sampler(corpus['config'].get('sampling_profile'))
    if source.get('global_answer_class_frequency_shaping_applied') is not (sampler is not None):
        raise ValueError('Completed corpus sampling verification changed')
    required = set(sampler.verification()) if sampler is not None else set()
    if corpus['config'].get('position_sampling_profile'):
        budgets = position_budgets(corpus['config']['task_position_budgets'])
        required.update(position_verification(budgets, source['task_position_selection'],
                                              corpus['config'].get('task_order_profile')))
    if corpus['config'].get('catalog_proof'):
        if verified.get('verified_native_source_catalog_reused') is not True:
            raise ValueError('Completed readback lacks its declared native catalog verification')
    if any(k not in verified or verified[k] != source.get(k) for k in required):
        raise ValueError('Completed readback sampling, task budgets or shortfall verification changed')
    if any(verified[k] != source[k] for k in verified.keys() & source.keys()):
        raise ValueError('Completed readback verification differs from the corpus producer')
    wanted = {corpus_path: corpus_binding}
    if set(_bindings(corpus['outputs'])) != {data / (s + '.jsonl') for s in ('train', 'validation', 'test')}:
        raise ValueError('Completed paper corpus must bind all three question files')
    for section in ('inputs', 'outputs'):
        for path, binding in _bindings(corpus[section]).items():
            if path in wanted and wanted[path] != binding:
                raise ValueError('Corpus source artifact bindings conflict')
            wanted[path] = binding
    if _bindings(proof['inputs']) != wanted:
        raise ValueError('Completed readback input bindings differ from the exact current corpus')
    verification_path = completed_manifest.parent / 'verification.json'
    outputs = _bindings(proof['outputs'])
    if set(outputs) != {verification_path}:
        raise ValueError('Completed readback lacks its complete verification artifact')
    signatures = {corpus_path: corpus_signature, completed_manifest: proof_signature}
    artifacts = {**wanted, **outputs}
    if any(path in wanted and wanted[path] != binding for path, binding in outputs.items()):
        raise ValueError('Completed readback has conflicting input/output bindings')
    for path, binding in artifacts.items():
        before = file_signature(path)
        if before[2] != binding['bytes'] or digest(path) != binding['sha256'] or file_signature(path) != before:
            raise ValueError(f'Completed native readback artifact changed: {path}')
        signatures.setdefault(path, before)
    if json.loads(verification_path.read_text()) != verified:
        raise ValueError('Completed readback verification artifact differs from its manifest')
    result = VerifiedPaperReadback(data, deepcopy(corpus), deepcopy(verified), proof_binding['sha256'],
                                 tuple(dict.fromkeys([completed_manifest, *artifacts])), signatures)
    result.check_unchanged()
    return result
